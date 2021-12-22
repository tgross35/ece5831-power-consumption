"""Module to load data for our neural network"""

import json
from datetime import datetime, timedelta

import numpy as np

from crystalball.models import CombinedData, InputRegionInfo, session
from crystalball.namesgenerator import get_random_name

SET_DIRECTORY = "data/sets"
# Where we split training and test data, use the last year for the split
TRAINING_DATA_SPLIT_DATE = datetime.fromisoformat("2016-11-30 00:00:00")
HISTORICAL_HOURS = 48
FUTURE_HOURS = 48

# use to valiate our ranges
START_END_TDIFF = timedelta(hours=HISTORICAL_HOURS + FUTURE_HOURS - 1)
START_NOW_TDIFF = timedelta(hours=HISTORICAL_HOURS - 1)

# input length has 3 elements for history (temp, description, power),
# two for future (temp and description) and 5 for datetime + lat/longitude
# future only has output power
INPUT_LENGTH = HISTORICAL_HOURS * 3 + FUTURE_HOURS * 2 + 5
OUTPUT_LENGTH = FUTURE_HOURS


def get_file_name():
    """Give our files a time-based name so they're easy to identify"""
    daytime = datetime.now().strftime("%d%H%M")
    return f"{SET_DIRECTORY}/{daytime}"


def process_dataset(data: list) -> tuple:
    """Turn our query into something that tf can use"""
    # Alright, so we want to extract two separate things: an "input" and an "output"
    # We pretend we are standing at a point in the dataset, call it "now"
    # Our input dataset then includes:
    # - information about now (day of year, hour, etc)
    # - hourly information about past power usage (up to and including now)
    # - hourly past weather information (up to and including now)
    # - future weather information (starting at now + 1 hour)
    # The goal is to be able to predict power usage for the next amount of time (1 day)
    # So, our output will be:
    # - hourly future power consumption

    nowindex = HISTORICAL_HOURS - 1
    maxdataindex = len(data) - 1 - FUTURE_HOURS - HISTORICAL_HOURS

    input_data = np.zeros((maxdataindex, INPUT_LENGTH))
    output_data = np.zeros((maxdataindex, OUTPUT_LENGTH))

    in_working = np.zeros(INPUT_LENGTH)
    out_working = np.zeros(OUTPUT_LENGTH)

    # Track where we are in arrays
    inout_index = 0
    inwork_index = 0
    outwork_index = 0
    skip_count = 0

    print(f"Data has {len(data)} elements")

    # Loop through the data in the range that we can get 48 (HISTORICAL_HOURS) before and
    # 24 (FUTURE_HOURS) after
    for i in range(0, maxdataindex):
        inwork_index = 0
        outwork_index = 0

        if i % 10000 == 0:
            print(f"At iteration {i}")

        # Collect this range into working info
        working_end = i + HISTORICAL_HOURS + FUTURE_HOURS
        working_set = data[i:working_end]
        now = working_set[nowindex]

        # Throw out data where we don't have a complete time set
        if working_set[0].dtime + START_END_TDIFF != working_set[-1].dtime:
            skip_count += 1
            continue

        if working_set[0].dtime + START_NOW_TDIFF != now.dtime:
            skip_count += 1
            continue

        # Load in things that don't change with time
        in_working[0] = now.year
        in_working[1] = now.dayofweek
        in_working[2] = now.hour
        in_working[3] = now.latitude
        in_working[4] = now.longitude
        inwork_index = 5

        # We are only going to look at temperature
        for w in working_set:
            tmp = inwork_index
            in_working[inwork_index] = w.weather_description_id
            inwork_index += 1
            in_working[inwork_index] = w.temperature
            inwork_index += 1
            assert inwork_index == tmp + 2

            # Append current & past to inputs, future to outputs
            if w.dtime <= now.dtime:
                in_working[inwork_index] = w.power_mw
                inwork_index += 1
                assert inwork_index == tmp + 3
            else:
                out_working[outwork_index] = w.power_mw
                outwork_index += 1
                assert inwork_index == tmp + 2

        # Just double check that we didn't mix anything up or miss something
        assert inwork_index == INPUT_LENGTH
        assert outwork_index == OUTPUT_LENGTH

        input_data[inout_index] = in_working
        output_data[inout_index] = out_working
        inout_index += 1

    remove_indicies = [x for x in range(inout_index, len(input_data))]

    input_data = np.delete(input_data, remove_indicies, axis=0)
    output_data = np.delete(output_data, remove_indicies, axis=0)

    print(f"Finished one dataset, skipped {skip_count}")

    return (input_data, output_data)


def get_data():
    """Get data for tensorflow"""
    print("Getting data")

    # Get region IDs in use
    region_ids = session.query(InputRegionInfo).all()
    region_ids = [r.id for r in region_ids]

    train_set_in = None
    train_set_out = None
    test_set_in = None
    test_set_out = None

    for region_id in region_ids:
        print(f"Working on region {region_id}")
        q = (
            session.query(CombinedData)
            .filter(CombinedData.region_id == region_id)
            .order_by(CombinedData.dtime)
        )

        training_region_data = q.filter(
            CombinedData.dtime < TRAINING_DATA_SPLIT_DATE
        ).all()

        test_region_data = q.filter(
            CombinedData.dtime >= TRAINING_DATA_SPLIT_DATE
        ).all()

        # Skip regions with no data
        if (
            training_region_data is None
            or test_region_data is None
            or len(training_region_data) == 0
            or len(test_region_data) == 0
        ):
            continue

        print("Creating region %i training set" % region_id)
        tmp_training_set = process_dataset(training_region_data)

        print("Creating region %i test set" % region_id)
        tmp_test_set = process_dataset(test_region_data)

        # First iteration, just set equal
        if train_set_in is None:
            train_set_in = tmp_training_set[0]
            train_set_out = tmp_training_set[1]
            test_set_in = tmp_test_set[0]
            test_set_out = tmp_test_set[1]
        else:
            train_set_in = np.append(train_set_in, tmp_training_set[0], axis=0)
            train_set_out = np.append(train_set_out, tmp_training_set[1], axis=0)
            test_set_in = np.append(test_set_in, tmp_test_set[0], axis=0)
            test_set_out = np.append(test_set_out, tmp_test_set[1], axis=0)

        print(f"Finished working on region {region_id}")

    # We've come all this way - better save our hard work

    fname = get_file_name()
    np.savez(
        fname,
        train_set_in=train_set_in,
        train_set_out=train_set_out,
        test_set_in=test_set_in,
        test_set_out=test_set_out,
    )
    fname_shape = f"{fname}.shape"
    shapes = {
        "train_set_in": train_set_in.shape,
        "train_set_out": train_set_out.shape,
        "test_set_in": test_set_in.shape,
        "test_set_out": test_set_out.shape,
    }
    with open(fname_shape, "w") as f:
        json.dump(shapes, f)

    print(f"Saved numpy arrays to {fname}")


def run_collection():
    """Create the model and run it"""
    get_data()
