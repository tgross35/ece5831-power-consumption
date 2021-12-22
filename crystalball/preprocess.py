"""Put the data into a usable table"""

from crystalball.models import create_if_not_exists, engine, session


def compress_weather_data():
    """Cheat and use raw SQL to flatten our imported weather data"""

    stmt = """
    INSERT INTO compressed_weather_data (
        weather_city_id,
        dtime,
        year,
        dayofyear,
        dayofweek,
        hour,
        humidity,
        pressure,
        temperature,
        weather_description,
        weather_description_id,
        wind_direction,
        wind_speed
        )
    SELECT
        weather_city_id,
        dtime,
        year,
        dayofyear,
        dayofweek,
        hour,
        GROUP_CONCAT(humidity),
        GROUP_CONCAT(pressure),
        GROUP_CONCAT(temperature_kelvin) - 273.15,
        GROUP_CONCAT(weather_description),
        GROUP_CONCAT(weather_description_id),
        -- Only take first value, sometimes there are multiple
        -- SUBSTR(GROUP_CONCAT(weather_description_id), 0, INSTR(GROUP_CONCAT(weather_description_id), ',')),
        GROUP_CONCAT(wind_direction),
        GROUP_CONCAT(wind_speed)
        FROM input_weather_data 
        GROUP BY weather_city_id, dtime ; 
    """

    engine.execute(stmt)

    # Fix issue where sometimes weather descriptions actually have something to concat
    stmt_fix_comma = """
    UPDATE compressed_weather_data 
    SET weather_description_id = SUBSTR(weather_description_id, 0, INSTR(weather_description_id, ','))
    WHERE weather_description_id LIKE '%,%';
    """

    engine.execute(stmt_fix_comma)


def combine_weather_and_power_data():
    """Put it all together into one table here.

    Make sure we only take columns that overlap."""
    stmt = """
    INSERT INTO combined_data (
        dtime,
        year,
        dayofyear,
        dayofweek,
        hour,
        region_id,
        code,
        weather_city_id,
        latitude,
        longitude,
        power_mw,
        humidity,
        pressure,
        temperature,
        weather_description,
        weather_description_id,
        wind_direction,
        wind_speed
    )
    SELECT 
        ipd.dtime,
        ipd.`year`,
        ipd.dayofyear,
        ipd.dayofweek,
        ipd.`hour`,
        ipd.region_id, 
        iri.code,
        cwd.weather_city_id,
        iri.latitude,
        iri.longitude,
        ipd.power_mw, 
        cwd.humidity,
        cwd.pressure,
        cwd.temperature,
        cwd.weather_description,
        cwd.weather_description_id,
        cwd.wind_direction,
        cwd.wind_speed
    FROM
        input_power_data ipd
    INNER JOIN input_region_info iri ON
        ipd.region_id = iri.id
    INNER JOIN compressed_weather_data cwd ON
        (iri.weather_city_id = cwd.weather_city_id
            AND 
        ipd.`dtime` = cwd.`dtime`)
    ORDER BY
        ipd.dtime;
    """

    engine.execute(stmt)


def preprocess_data():
    create_if_not_exists()
    print("Compressing weather data")
    compress_weather_data()
    print("Done. Joining power and weather data")
    combine_weather_and_power_data()
    print("Done. Data is ready for processing.")
