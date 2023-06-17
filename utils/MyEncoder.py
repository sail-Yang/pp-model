import json
import datetime
from array import array


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, array):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def fandata_2_json(obj):
    return {
        "id": obj.id,
        "fan_id": obj.fan_id,
        "datatime": obj.datatime.strftime('%Y-%m-%d %H:%M:%S'),
        "windspeed": obj.windspeed,
        "prepower": obj.prepower,
        "winddirection": obj.winddirection,
        "temperature": obj.temperature,
        "humidity": obj.humidity,
        "pressure": obj.pressure,
        "ws": obj.ws,
        "power": obj.power,
        "yd15": obj.yd15
    }


def test_fandata_2_json(obj):
    return {
        "datatime": obj.datatime.strftime('%Y-%m-%d %H:%M:%S'),
        "windspeed": obj.windspeed,
        "prepower": obj.prepower,
        "winddirection": obj.winddirection,
        "temperature": obj.temperature,
        "humidity": obj.humidity,
        "pressure": obj.pressure,
        "ws": obj.ws,
        "power": obj.power,
        "yd15": obj.yd15
    }

