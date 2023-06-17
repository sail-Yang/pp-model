from datetime import datetime, timedelta


def getYesterDay():
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
