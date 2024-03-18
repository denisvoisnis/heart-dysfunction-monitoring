import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, date
from urllib.parse import quote_plus
from typing import Iterable, Union, Optional, Dict
import numpy as np
import pandas as pd
import pytz
from sqlalchemy import text, create_engine
from sqlalchemy.engine.base import Engine
import psycopg2
from psycopg2.extensions import connection
import yaml

Connection = None


# Local timezone
TZ = pytz.utc.__str__()
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')


## Logger functions

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    if not os.path.isdir(LOGS_DIR):
        os.mkdir(LOGS_DIR)
    logger = logging.Logger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)-15s [%(levelname)s]: %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = RotatingFileHandler(os.path.join(LOGS_DIR, name + '.log'), maxBytes=10 ** 7, backupCount=10)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def _log_header(script: Optional[str], function: Optional[str]) -> str:
    if script is not None and function is not None:
        msg = '[{}:{}] '.format(script, function)
    elif script is not None and function is None:
        msg = '[{}] '.format(script)
    elif script is None and function is not None:
        msg = '[{}] '.format(function)
    else:
        msg = ''
    return msg

def _log_range(date_from: Optional[Union[date, datetime]], date_to: Optional[Union[date, datetime]]) -> str:
    if date_from is not None and date_to is not None:
        msg = '[Range ({} : {})] '.format(date_from, date_to)
    elif date_from is not None and date_to is None:
        msg = '[{}] '.format(date_from)
    elif date_from is None and date_to is not None:
        msg = '[{}] '.format(date_to)
    else:
        msg = ''
    return msg

def log_to_database(engine: Union[Engine, connection],
            execution_time: datetime = datetime.now(),
            script: Optional[str] = None,
            function: Optional[str] = None,
            user_id: Optional[str] = None,
            date_from: Optional[Union[date, datetime]] = None,
            date_to: Optional[Union[date, datetime]] = None,
            description: Optional[str] = None,
            level:Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            datasource_id: Optional[str] = None):
    # use default logger if not set
    logger = logger or get_logger(__name__)
    header = _log_header(script, function)
    if engine is None:
        logger.error('{}Error while writing log to database: connection could not be initialized'.format(header))
        return
    # Normalize timezone, as event_date is stored using timestamptz
    execution_time = execution_time or datetime.now()
    if execution_time.tzinfo is not None:
        execution_time = execution_time.replace(tzinfo=None)
    execution_time = pytz.timezone(TZ).localize(execution_time).astimezone(pytz.utc).isoformat()

    log_stmt = """
    INSERT INTO dhealth."LogProcessing" ("EventDate", "Script", "Function", "UserId", "DateFrom", "DateTo", "Description", "Level", "DataSourceId")
    VALUES (:event_date, :script, :function, :user_id, :date_from, :date_to, :description, :level, :datasource_id);
    """
    values = {'event_date': execution_time,  'script': script, 'function': function,
        'user_id': user_id, 'date_from': date_from,  'date_to': date_to,
        'description': description, 'level': level, 'datasource_id': datasource_id}
    
    try:
        with engine.connect() as con:
            con.execute(statement = text(log_stmt), parameters = values)
    except Exception as e:
        logger.error("{}Error writing log to database: {}".format(header, e.__str__()))


def log_write(engine: Union[Engine, connection], **kwargs):
    logger = kwargs.get('logger') or get_logger(__name__)
    description = kwargs.get('description')
    date_from = kwargs.get('date_from')
    date_to = kwargs.get('date_to')
    user_id = kwargs.get('user_id')
    level = kwargs.get('level') or 'WARNING'
    log_to_database(engine, **kwargs)
    if description is None: return
    header = _log_header(kwargs.get('script'), kwargs.get('function'))
    if date_from is not None and date_to is not None:
        suff = _log_range(date_from, date_to)
    elif user_id is not None:
        suff = '[User {}]'.format(user_id)
    else: suff = ''
    msg = header + suff + description
    logging_funcs = {'INFO': logger.info, 'WARNING': logger.warning, 'ERROR': logger.error}
    logging_fun = logging_funcs.get(level or 'INFO') or logger.info
    logging_fun(msg)


### Generic data retrieve functions

def store_pandas_df(pg_engine: Union[Engine, connection], df: pd.DataFrame, tablename: str,
                    log_table_name: str=None, schema: str='dhealth', method='multi', **kwargs):
    user_id = kwargs.get('user_id')
    logger = kwargs.get('logger') or get_logger(__name__)
    header = _log_header(kwargs.get('script'), kwargs.get('function'))
    log_table_name = log_table_name or tablename
    if df is None or df.shape[0] == 0:
        kwargs['description'] = 'No records inserted into {} table'.format(log_table_name)
        kwargs['level'] = 'INFO'
        log_to_database(pg_engine, **kwargs)
        return
    try:
        df = df.replace({np.nan: None})
        df.to_sql(tablename, pg_engine, schema=schema, if_exists='append', index=False, method=method)
    except ValueError as e:
        if user_id is not None:
            header = '{}[User {}]'.format(header, user_id)
        logger.error('{} Error while saving {}: {}'.format(header, log_table_name, e.__str__()))


def store_periodical_df(pg_engine: Union[Engine, connection], df: pd.DataFrame, tablename: str,
                        date_from: Union[date, datetime], date_to: Union[date, datetime],
                        log_table_name: str=None, schema: str='dhealth', method='multi', **kwargs):
    # Override settings in kwargs with the mandatory params
    kwargs['date_from'] = date_from
    kwargs['date_to'] = date_to
    logger = kwargs.get('logger') or get_logger(__name__)
    header = _log_header(kwargs.get('script'), kwargs.get('function'))
    log_table_name = log_table_name or tablename
    rng = _log_range(date_from, date_to)
    stats_df = pd.DataFrame(data=[{'Schema': schema, 'TableName': tablename, 'DateFrom': date_from,
                                   'DateTo': date_to,'NumEntries': 0, 'UserId': kwargs.get('user_id')}])
    if df is None or df.shape[0] == 0:
        kwargs['description'] = 'No records inserted into {} table'.format(log_table_name)
        kwargs['level'] = 'INFO'
        log_to_database(pg_engine, **kwargs)
        stats_df.to_sql('InsertionStats', pg_engine, if_exists='append', index=False)
        return
    try:
        df.to_sql(tablename, pg_engine, schema=schema, if_exists='append', index=False, method=method)
        stats_df['NumEntries'] = df.shape[0]
        stats_df.to_sql('InsertionStats', pg_engine, if_exists='append', index=False)
    except ValueError as e:
        kwargs['description'] = 'Error while saving {}: {}'.format(log_table_name, e.__str__())
        kwargs['level'] = 'ERROR'
        log_to_database(pg_engine, **kwargs)
        logger.error('{}{}Error while saving {}: {}'.format(header, rng, log_table_name, e.__str__()))


### Connection initialization functions

# def create_msssql_connection(conn: Connection) -> Engine:
#     mssql_driver = 'ODBC Driver 17 for SQL Server'
#     port = conn.port or 1433
#     source_str = quote_plus('driver={{{}}};server={};port={};database={};uid={};pwd={};TDS_Version=8.0;'
#                         .format(mssql_driver, conn.host, port, conn.schema, conn.login, conn.password))
#     return create_engine("mssql+pyodbc:///?odbc_connect={}".format(source_str), fast_executemany=True)\
#         .execution_options(isolation_level="AUTOCOMMIT")


# def create_postgresql_connection(conn: Connection) -> Engine:
#     port = conn.port or 5432
#     return create_engine("postgresql+psycopg2://{}:{}@{}:{}/{}".format(conn.login, conn.password, conn.host, port, conn.schema),
#                           pool_pre_ping=True, pool_use_lifo=True, max_overflow=2)\
#         .execution_options(isolation_level="AUTOCOMMIT")


# def create_psycopg_connection(conn: Connection) -> connection:
#     port = conn.port or 5432
#     pg_conn_str = "host={} port={} dbname={} user={} password={}".format(conn.host, port, conn.schema, conn.login, conn.password)
#     return psycopg2.connect(pg_conn_str)



def read_credentials(path, cred_name ):
    credentials = yaml.load(open(path), Loader=yaml.SafeLoader)
    return credentials[cred_name]
    
 

def create_postgresql_connection(conn):
    port = conn[ 'port']
    login =  conn[ 'login'] 
    password = conn[ 'password']
    host =  conn[ 'host']
    schema = conn[ 'schema'] 
    return create_engine("postgresql+psycopg2://{}:{}@{}:{}/{}".format(login, password, host, port, schema),
                         pool_pre_ping=True, pool_use_lifo=True, max_overflow=2)\
        .execution_options(isolation_level="AUTOCOMMIT")

       
def get_sql_connection( db_name, conn):
    mssql_driver = conn['mssql_driver'] 
    port = conn['port'] 
    SERVER = conn['SERVER'] 
    # db_name='Measurements'
    UID = conn['UID'] 
    PWD = conn['PWD'] 
    
    source_str = quote_plus('driver={{{}}};server={};port={};database={};uid={};pwd={};TDS_Version=8.0;'
                            .format(mssql_driver, SERVER, port,
                                    db_name, UID, PWD ))
    try:
        conn = create_engine("mssql+pyodbc:///?odbc_connect={}".format(source_str), fast_executemany=True)
    except Exception as e:
        conn = None
    return conn
        
        
        

