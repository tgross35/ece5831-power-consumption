import csv
import json
import os
from datetime import datetime
from glob import glob

import pytz

from crystalball.models import (
    InputPowerData,
    InputRegionInfo,
    InputWeatherData,
    WeatherCity,
    create_if_not_exists,
    session,
)

# glob data folder
# for each csv

REGIONS_JSON = "raw/regions.json"
POWER_GLOB = "raw/power/*.csv"
WEATHER_GLOB = "raw/weather/*.csv"
# first date of power data for most sources
START_DATE = datetime.fromisoformat("2013-01-01 00:00:00").astimezone(pytz.UTC)
# last date of weather data
END_DATE = datetime.fromisoformat("2017-11-30 00:00:00").astimezone(pytz.UTC)


def load_region():
    with open(REGIONS_JSON, "r") as f:
        regions = json.load(f)

    # Load in the regions
    for region in regions["regions"]:
        if region.get("process"):
            weather_city = region.get("weather_city")

            # Create cities that don't exist
            wcobj = (
                session.query(WeatherCity)
                .filter(WeatherCity.name == weather_city)
                .first()
            )
            if wcobj is None:
                wcobj = WeatherCity.create(name=weather_city)

            InputRegionInfo.create(
                code=region.get("code"),
                description=region.get("description"),
                city=region.get("city"),
                timezone=region.get("timezone"),
                weather_city_id=wcobj.id,
                latitude=region.get("latitude"),
                longitude=region.get("longitude"),
            )


def load_power():
    power_glob = glob(POWER_GLOB)
    power_glob.sort()
    # Load in all the power data
    for fname in power_glob:
        shortname = fname.split(os.sep)[-1]
        code = shortname.split("_")[0].upper()
        region_obj = (
            session.query(InputRegionInfo).filter(InputRegionInfo.code == code).first()
        )

        # Exclude regions we don't want to work with
        if region_obj is None:
            continue
        local_tz = pytz.timezone(region_obj.timezone)

        displayed_years = []

        with open(fname, "r") as f:
            reader = csv.reader(f)
            insertlist = []

            # Skip the first row
            next(reader)

            print("Loading file %s" % shortname)
            for row in reader:
                # Datetimes are in ISO formats, in local time. Change it to UTC.
                date_local = datetime.fromisoformat(row[0])
                date = local_tz.localize(date_local).astimezone(pytz.UTC)

                # Don't bother loading anything outside of our period of interest
                if date < START_DATE or date > END_DATE:
                    continue

                # Send a status message if we're at a first of a year
                if date.timetuple().tm_yday == 1:
                    year = date.year
                    if year not in displayed_years:
                        print("Power at %i" % year)
                        displayed_years.append(year)

                insertlist.append(
                    InputPowerData(
                        region_id=region_obj.id,
                        dtime=date,
                        year=date.year,
                        dayofyear=date.timetuple().tm_yday,
                        dayofweek=date.weekday(),
                        hour=date.hour,
                        power_mw=int(float(row[1])),
                    )
                )

            session.bulk_save_objects(insertlist)
            session.commit()

    print("Finished loading all power files")


def load_weather():
    weather_glob = glob(WEATHER_GLOB)
    weather_glob.sort()

    weather_cities = session.query(WeatherCity).all()

    # What columns we get from which files
    file_col_map = {
        "pressure.csv": "pressure",
        "weather_description.csv": "weather_description",
        "wind_speed.csv": "wind_speed",
        "humidity.csv": "humidity",
        "temperature.csv": "temperature_kelvin",
        "wind_direction.csv": "wind_direction",
    }

    # Assign int values to weather descriptions
    weather_description_map = {
        "broken clouds": 1,
        "drizzle": 2,
        "dust": 3,
        "few clouds": 4,
        "fog": 5,
        "freezing rain": 6,
        "haze": 8,
        "heavy intensity drizzle": 8,
        "heavy intensity rain": 9,
        "heavy snow": 10,
        "light intensity drizzle": 11,
        "light intensity shower rain": 12,
        "light rain": 13,
        "light rain and snow": 14,
        "light snow": 15,
        "mist": 16,
        "moderate rain": 17,
        "overcast clouds": 18,
        "proximity shower rain": 19,
        "proximity thunderstorm": 20,
        "proximity thunderstorm with drizzle": 21,
        "proximity thunderstorm with rain": 22,
        "scattered clouds": 23,
        "sky is clear": 24,
        "smoke": 25,
        "snow": 26,
        "squalls": 27,
        "thunderstorm": 28,
        "thunderstorm with drizzle": 29,
        "thunderstorm with heavy rain": 30,
        "thunderstorm with light drizzle": 31,
        "thunderstorm with light rain": 32,
        "thunderstorm with rain": 33,
        "very heavy rain": 34,
    }

    # Loop through files
    for fname in weather_glob:
        shortname = fname.split(os.sep)[-1]
        col_name = file_col_map.get(shortname)
        if col_name is None:
            continue

        print("Loading file %s" % shortname)

        displayed_years = []

        with open(fname, "r") as f:
            reader = csv.DictReader(f)
            insertlist = []

            for row in reader:
                # Here, dates are in UTC
                date = datetime.fromisoformat(row["datetime"]).astimezone(pytz.UTC)

                # Don't bother loading anything outside of our period of interest
                if date < START_DATE or date > END_DATE:
                    continue

                # Send a status message if we're at a first of a year
                if date.timetuple().tm_yday == 1:
                    year = date.year
                    if year not in displayed_years:
                        print("Weather at %i" % year)
                        displayed_years.append(year)

                for city in weather_cities:
                    # Trickery to get through blank values
                    data = row[city.name]
                    if data == "":
                        data = "0"

                    insertvals = {
                        "dtime": date,
                        "year": date.year,
                        "dayofyear": date.timetuple().tm_yday,
                        "dayofweek": date.weekday(),
                        "hour": date.hour,
                        "weather_city_id": city.id,
                        col_name: data,
                    }

                    # If we have a description, get the ID too. Treat two-item strings
                    # as first item only
                    if col_name == "weather_description":
                        insertvals["weather_description_id"] = weather_description_map[
                            data.split(",")[0]
                        ]

                    insertlist.append(InputWeatherData(**insertvals))

            session.bulk_save_objects(insertlist)
            session.commit()


def load_data(task: str = "all"):
    """Load all our data in, or some of it as specified."""

    print("Creating tables")
    create_if_not_exists()
    print("Created, loading files")

    load_region()

    if task == "weather":
        load_weather()
    elif task == "power":
        load_power()
    elif task == "all":
        load_power()
        load_weather()
