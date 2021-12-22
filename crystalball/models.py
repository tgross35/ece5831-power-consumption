from typing import Type, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.orm.query import Query
from sqlalchemy.sql.sqltypes import Float

T = TypeVar("T")

Base = declarative_base()
engine = create_engine("sqlite:///data/data.sqlite")
Session = sessionmaker(bind=engine)
session = Session()


class CRUDMixin:
    """Mixin that adds convenience methods for CRUD (create, read, update, delete) operations."""

    @classmethod
    def create(cls: Type[T], **kwargs) -> T:
        """Create a new record and save it the database."""
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        """Update specific fields of a record."""
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        if commit:
            return self.save()
        return self

    def save(self, commit=True):
        """Save the record."""
        session.add(self)
        if commit:
            session.commit()
        return self

    def delete(self, commit: bool = True) -> None:
        """Remove the record from the database."""
        session.delete(self)
        if commit:
            return session.commit()
        return


class Model(CRUDMixin, Base):
    """Base model class that includes CRUD convenience methods."""

    __abstract__ = True


class PKMixin:
    """Base model class that includes CRUD convenience methods, plus adds a 'primary key' column named ``id``."""

    __abstract__ = True
    id = Column(Integer, primary_key=True)

    @classmethod
    def get_by_id(cls: Type[T], record_id) -> T:
        """Get record by ID."""
        return cls.query.get(int(record_id))


class TimeProcessMixin:
    """Class mixin for everything that works with dates."""

    dtime = Column(DateTime, index=True)
    year = Column(Integer, index=True)
    dayofyear = Column(Integer, index=True)
    dayofweek = Column(Integer, index=True)
    hour = Column(Integer, index=True)


class InputRegionInfo(Model, PKMixin):
    """Short table with information about regions."""

    __tablename__ = "input_region_info"
    code = Column(String, index=True)
    description = Column(String)
    city = Column(String)
    weather_city_id = Column(
        ForeignKey("weather_city_info.id"), index=True
    )  # Closest city we have weather data for
    timezone = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)


class WeatherCity(Model, PKMixin):
    """A single city where we have weather data."""

    __tablename__ = "weather_city_info"
    name = Column(String)


class InputPowerData(Model, PKMixin, TimeProcessMixin):
    """All power data for all regions."""

    __tablename__ = "input_power_data"
    region_id = Column(ForeignKey("input_region_info.id"), index=True)
    power_mw = Column(Float)

    region = relationship("InputRegionInfo", backref="power_data")

    def __repr__(self) -> str:
        return f"{self.dtime} {self.region.name} {self.power_mw}"


class InputWeatherData(Model, PKMixin, TimeProcessMixin):
    """All weather data for all regions."""

    __tablename__ = "input_weather_data"
    weather_city_id = Column(
        ForeignKey("weather_city_info.id"), index=True
    )  # Closest city we have weather data for

    humidity = Column(Float)
    pressure = Column(Float)
    temperature_kelvin = Column(Float)
    weather_description = Column(String)
    weather_description_id = Column(Integer)  # weather description
    wind_direction = Column(Float)
    wind_speed = Column(Float)

    city = relationship("WeatherCity", backref="weather_data")

    def __repr__(self) -> str:
        return f"{self.dtime} {self.city.name} {self.temperature}"


class CompressedWeatherData(Model, PKMixin, TimeProcessMixin):
    """All weather data for all regions, compressed vertically."""

    __tablename__ = "compressed_weather_data"
    weather_city_id = Column(
        ForeignKey("weather_city_info.id"), index=True
    )  # Closest city we have weather data for

    humidity = Column(Float)  # percent
    pressure = Column(Float)  # in millibar
    temperature = Column(Float)  # in celsius
    weather_description = Column(String)
    weather_description_id = Column(Integer)  # weather description
    wind_direction = Column(Float)  # bearing
    wind_speed = Column(Float)

    city = relationship("WeatherCity", backref="weather_data_processed")

    def __repr__(self) -> str:
        return f"{self.dtime} {self.city.name} {self.temperature} {self.weather_description}"


class CombinedData(Model, PKMixin, TimeProcessMixin):
    """All input data in one place to save time from joins in practice"""

    __tablename__ = "combined_data"

    region_id = Column(ForeignKey("input_region_info.id"), index=True)
    code = Column(String, index=True)
    weather_city_id = Column(ForeignKey("weather_city_info.id"), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    power_mw = Column(Float)
    humidity = Column(Float)  # percent
    pressure = Column(Float)  # in millibar
    temperature = Column(Float)  # in celsius
    weather_description = Column(String)  # weather description
    weather_description_id = Column(Integer)  # weather description
    wind_direction = Column(Float)  # bearing
    wind_speed = Column(Float)

    region = relationship("InputRegionInfo", backref="combined_data")
    city = relationship("WeatherCity", backref="combined_data")

    def __repr__(self) -> str:
        return f"{self.city.name} {self.dtime} {self.temperature} {self.power_mw}"


def create_if_not_exists():
    """Create all tables"""
    Base.metadata.create_all(engine)
