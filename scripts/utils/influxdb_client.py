from dotenv import load_dotenv
from influxdb_client_3 import InfluxDBClient3

import os

load_dotenv()

def is_valid_env():
    return os.getenv('INFLUXDB_TOKEN') and os.getenv('INFLUXDB_HOST') and os.getenv('INFLUXDB_RAW_BUCKET')

def get_client() -> InfluxDBClient3:
    """
    Create and return an InfluxDB client instance.
    This function loads the InfluxDB connection parameters from environment variables.
    It raises a ValueError if the required environment variables are not set.

    Returns:
        InfluxDBClient3: An instance of the InfluxDB client.
    Raises:
        ValueError: If the required environment variables are not set.
    """
    token = os.getenv('INFLUXDB_TOKEN')
    host = os.getenv('INFLUXDB_HOST')
    database = os.getenv('INFLUXDB_RAW_BUCKET')

    if not is_valid_env():
        raise ValueError('Please set the environment variables INFLUXDB_TOKEN, INFLUXDB_HOST, and INFLUXDB_RAW_BUCKET')
    
    client = InfluxDBClient3(host=host, database=database, token=token)

    return client
