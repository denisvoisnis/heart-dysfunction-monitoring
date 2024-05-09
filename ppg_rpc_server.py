##!/usr/bin/env python
import pika
import os
import numpy as np
# import time
from pika.exchange_type import ExchangeType
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
import json
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId, BSON
from heart_PA_ppg import heart_PA_ppg
from heart_PA_ecg import heart_PA_ecg
from utils_d import get_logger, create_postgresql_connection, log_write, read_credentials
from gridfs import GridFS
__location__ = str(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__))))) + "/config.yaml"

logger = get_logger('Heart Dysfunction Monitoring PPG Rabbit MQ', level=logging.DEBUG)
log_args = {
    'script': "Heart_Dysfunction_Monitoring_PPG",
    'level': 'ERROR',
    'logger': logger
}


def init_pg_connection():
    path = __location__  
    hadas = read_credentials( path = path, cred_name = 'hadas')
    try:
        pg_conn = create_postgresql_connection(hadas)
    except Exception as e:
        logging.error("Error while connecting to HADAS database: " + e.__str__())
        raise e
    return pg_conn

def create_mongo_conection():
    try:
        path = __location__
        mongo_db = read_credentials( path = path, cred_name = 'mongo')
        client = MongoClient(mongo_db['credentials'])
        collections= client[mongo_db['database']]
    except Exception as e:
        logging.error(f"Error while connecting to Mongo {mongo_db['database'] } database: " + e.__str__())
        raise e
    return collections

def get_data_from_mongo( UserId, jsonId, collections):
    fs = GridFS( create_mongo_conection(), collection="H10Data")
    _id = ObjectId(jsonId)
    bytes_data = fs.get(_id).read()
    data = BSON(bytes_data).decode()
    return data


def manage_msg_body(body):
    log_args['function'] = 'manage_msg_body'
    log_args['datasource_id'] = None
    log_args['level'] = 'ERROR'
    Error_msg = []
    pg_conn = init_pg_connection()
    try:
        msg_body =json.loads(body)
    except Exception as e:
        Error_msg = "Could not load message json:" + e.__str__()
        log_write(pg_conn , description= Error_msg , **log_args)
        return  (None, )*7 + (Error_msg, )
    try:
        messageId = msg_body['messageId']
        conversationId = msg_body['conversationId']
        messageType = ['urn:message:Events.Measurements:HeartDysfunctionReportCreated']
        sentTime = msg_body['sentTime'] 
        userId = msg_body['message']['userId']
        jsonId =  msg_body['message']['id']
        sourceAddress =  msg_body['sourceAddress']
    except Exception as e:
        Error_msg = "Could not get data from message:" + e.__str__()
        log_write(pg_conn , description= Error_msg , **log_args)
        return  (None, )*7 + (Error_msg, )
    
    pg_conn.dispose()
    return  messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg

def process_jsons(body):
    messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg = manage_msg_body(body)
    if len(Error_msg) > 10:
        return None, messageId, conversationId, messageType, sentTime, userId, jsonId, Error_msg
    
    log_args['function'] = 'process_jsons'
    log_args['datasource_id'] = None
    log_args['level'] = 'ERROR'
    pg_conn = init_pg_connection()
    collection_data  = [],[]
    collections = create_mongo_conection()
    try: 
        collection_data = get_data_from_mongo( userId,jsonId, collections) 
    except Exception as e:
        log_write(pg_conn , description="Error in retreaving data from mongo db" + e.__str__(), **log_args)
        return  None, messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg

    if len(collection_data) == 0 :
        log_args['level'] = 'WARNING'
        Error_msg=f"Warning data is empty: json counts { len(collection_data)} for user {userId} and Json ID {jsonId} "
        log_write(pg_conn , description=Error_msg, **log_args)
        return  None, messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg
    pg_conn.dispose()
       
    return collection_data, messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg


def calculate_heart_PA_ecg(collection_data_j):
    Error_msg = []
   
    metadata_ecg = {'acc_fs': 200,
     'position': collection_data_j['position'],
     'age': collection_data_j['age'],
     'gender': collection_data_j['gender'],
     'step_length': np.nan, 
     'height': collection_data_j['height'],
     'sub_id': 7,
     'filename': '',
     'recovery_time': 60,
     'rest_time': 30,
     'max_stop': 5,
     'min_walking_duration': 60,
     'minimal_walk_test_duration': 300,
     'max_walking_steps_recovery': 16,
     'active_minute_steps': 60,
     'max_steps_during_rest': 5,
     'fast_phase_window': 30}
   
    
    log_args['datasource_id'] = None
    log_args['level'] = 'ERROR'
    pg_conn = init_pg_connection()
    
    chest_keys= ['timestamp_acc_chest','x_chest','y_chest','z_chest']
    is_chest_there = [i for i in [*collection_data_j.keys()] if i in chest_keys]
    
    if len(is_chest_there) == len(chest_keys):
        acc_chest = pd.DataFrame({
            'timestamp' : collection_data_j['timestamp_acc_chest'],
            'x' : collection_data_j['x_chest'],
            'y' : collection_data_j['y_chest'],
            'z' : collection_data_j['z_chest']
            }
        )
        rr = pd.DataFrame( { 'timestamp' : collection_data_j['timestamp_rr'],
                       'rrms' : collection_data_j['rr'] } )
                       
        try:
            log_args['function'] = 'heart_PA_ecg'
            ecg_instance = heart_PA_ecg(acc_chest, rr, metadata_ecg)
        except Exception as e:
            Error_msg=f'Error while calculating ECG: {e.__str__()}'
            log_write(pg_conn , description=Error_msg, **log_args)
            return [],[], Error_msg
        
        try:
            log_args['function'] = 'ecg_instance.perform_analysis'
            df_sig_results_ecg, df_walking_bouts_results_ecg  = ecg_instance.perform_analysis()
        except Exception as e:
            # print('log_args: ' + str(log_args))
            Error_msg = f'Error while calculating ECG: {e.__str__()}'
            log_write(pg_conn , description = Error_msg, **log_args)
            return [],[], Error_msg
        
    elif  0 < len(is_chest_there) < len(chest_keys):
        missing_column = [i for i in chest_keys if i not in is_chest_there]
        Error_msg=f'Missing column in mongo collection, cant calcualte heart_PA_ecg. Lost {missing_column}, recerved columns: {is_chest_there}'
        log_write(pg_conn ,  description=Error_msg, **log_args)
        return [],[], Error_msg
    
    elif len(is_chest_there) == 0:
        log_args['level'] = 'DEBUG'
        Error_msg='No data to calculate heart_PA_ecg'
        log_write(pg_conn ,  description=Error_msg, **log_args)   
        pg_conn.dispose()
        return [],[], Error_msg
    pg_conn.dispose()
    return df_sig_results_ecg.to_json(orient='records'), df_walking_bouts_results_ecg.to_json(orient='records'), Error_msg

    ##############################################################################################

def calculate_heart_PA_ppg(collection_data_j):
    Error_msg = []
    metadata_ppg = {'acc_fs': 50,
     'ppg_fs': 135,
     'position': collection_data_j['position'],
     'age': collection_data_j['age'],
     'gender': collection_data_j['gender'],
     'height': collection_data_j['height'],
     'step_length': 'nan',
     'sub_id': 7,
     'filename': ''}
    
    log_args['datasource_id'] = None
    log_args['level'] = 'ERROR'
    pg_conn = init_pg_connection()
    
    arm_keys= ['timestamp_acc_arm','x_arm','y_arm','z_arm', 'ppg', 'subject_id']
    is_arm_there = [i for i in [*collection_data_j.keys()] if i in arm_keys]

    
    if len(is_arm_there) == len(arm_keys):
        acc_arm = pd.DataFrame({
            'timestamp' : collection_data_j['timestamp_acc_arm'],
            'x' : collection_data_j['x_arm'],
            'y' : collection_data_j['y_arm'],
            'z' : collection_data_j['z_arm']
            }
        )
        ppg = pd.DataFrame({ 'timestamp' : collection_data_j['timestamp_ppg'],
                       'ppg0' : collection_data_j['ppg'] } )
                       
        try:
            log_args['function'] = 'heart_PA_ppg'
            ppg_instance = heart_PA_ppg(acc_arm, ppg, metadata_ppg)
        except Exception as e:
            Error_msg=f'Error while calculating PPG: {e.__str__()}'
            log_write(pg_conn , description=Error_msg , **log_args)
            pg_conn.dispose()
            return [],[], Error_msg

        try:
            log_args['function'] = 'ppg_instance.perform_analysis'
            df_sig_results, df_walking_bouts_results  = ppg_instance.perform_analysis()
        except Exception as e:
            Error_msg=f'Error while calculating PPG: {e.__str__()}'
            log_write(pg_conn ,description=Error_msg , **log_args)
            pg_conn.dispose()
            return [],[], Error_msg

    elif  0 < len(is_arm_there) < len(arm_keys):
        missing_column = [i for i in arm_keys if i not in is_arm_there]
        Error_msg=f'Missing column in mongo collection, cant calcualte heart_PA_ppg. Lost {missing_column}, recerved columns: {is_arm_there}'
        log_write(pg_conn ,  description=Error_msg, **log_args)   
        return [],[], Error_msg
    
    elif len(is_arm_there) == 0:
        log_args['level'] = 'DEBUG'
        Error_msg='No data to calculate heart_PA_ecg'
        log_write(pg_conn ,   description=Error_msg, **log_args)   
        pg_conn.dispose()
        return [],[], Error_msg
    
    pg_conn.dispose()
    return df_sig_results.to_json(orient='records'), df_walking_bouts_results.to_json(orient='records'), Error_msg

 
def calculate_result(body):
    log_args['datasource_id'] = None
    log_args['level'] = 'ERROR'
    log_args['function'] = 'calculate_result'
    data = dict()

    pg_conn = init_pg_connection()
    collection_data, messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg =  process_jsons(body)
    if len(Error_msg) > 1:
        data[ jsonId ] = {
            'UserId' :userId,
            'df_sig_results' : None,
            'df_walking_bouts_results' : None,
            'position' : None,
            'messageId' : messageId,
            'conversationId' : conversationId,
            'device' : None,
            'testType' : None,
            'Error' :  Error_msg
        }
        data=json.dumps(data)    
        return data

    
    collection_data_j = collection_data
    df_sig_results =  []
    df_walking_bouts_results = []
    position = collection_data_j['position']
    device = 'polar'
    testType = collection_data_j['test_type']
    Error_msg = []
    
    if position == 'chest':
        print('chest')
        df_sig_results, df_walking_bouts_results, Error_msg = calculate_heart_PA_ecg(collection_data_j)
    elif position == 'wrist':
        print('wrist')
        df_sig_results, df_walking_bouts_results, Error_msg = calculate_heart_PA_ppg(collection_data_j)
    data = {
        'message': {
             '_id':  jsonId,
            'UserId': userId,
            'df_sig_results' : df_sig_results,
            'df_walking_bouts_results' : df_walking_bouts_results,
            'position' : position,
            'messageId' : messageId,
            'conversationId' : conversationId,
            'device' : device,
            'testType' : testType,
            'Error' :  Error_msg
        },        "messageType": messageType
              } 
    
    try:
        data=json.dumps(data)  
        print(data)
    except Exception as e:
        log_args['level'] = 'ERROR'
        Error_msg=f'Could not serialize dictionary into string: {e.__str__()}'
        log_write(pg_conn ,  description=Error_msg, **log_args)   
        pg_conn.dispose()
        return Error_msg
    return data

def answear_to_msg(ch, data):
    exchange_name='Events.Recommendations.HeartDysfunctionReportCreated'
    ch.exchange_declare( exchange= exchange_name ,  exchange_type= ExchangeType.fanout, durable = True)
    queue_name = 'HeartDysfunctionReportCreated'
    reply_queue = ch.queue_declare(queue = queue_name , durable = True)
    properties=pika.BasicProperties(
        reply_to=reply_queue.method.queue,
        )
    ch.queue_bind(queue_name, exchange_name)

    try:
        ch.basic_publish(exchange_name,
                         routing_key=properties.reply_to,
                         body=data,
                         properties=pika.BasicProperties(
                            delivery_mode = pika.DeliveryMode.Persistent
                            ) )

    except Exception as e:
        pg_conn = init_pg_connection()
        log_write(pg_conn , description="Error while publishing msg in rabbitMQ: " + e.__str__(), **log_args)


def on_request_message_received(ch, method, properties, body):
    messageId, conversationId, messageType, sentTime, userId, jsonId, sourceAddress, Error_msg = manage_msg_body(body)
    ExtraInfo = dict({
        'sentTime' : sentTime,
        'messageId' : messageId,
        'sourceAddress' : sourceAddress
        })
    data = calculate_result(body) 
    pg_conn = init_pg_connection()
    try:
        answear_to_msg(ch, data)
    except Exception as e:
        log_args['function'] = 'answear_to_msg'
        log_write(pg_conn , description="Faild to answear to message: " + e.__str__(), **log_args)
    try:
        d = { 'TimeInserted': datetime.now(tz=timezone.utc),
             'UserId' : userId,
             'DateFromUtc' : None,
             'DateToUtc' : None,
             'DigitalPrescription' :None,
             'MinutesRequired' : None,
             'Message': str(body) ,
             'MessageArgs':  None,
             'MessageType' : messageType,
             'ExtraInfo': ExtraInfo,
             'DataToSend' : str(data) }

        
        all_data_log = pd.DataFrame(d,  index=[0])
        pd.DataFrame.from_dict(d, orient='index')
        all_data_log.to_sql('PhysicalActivityReports', con=pg_conn, schema='dhealth', if_exists='append', index=False, method='multi')
    except Exception as e:
        log_args['function'] = 'all_data_log_ppg'
        log_write(pg_conn , description="Error while writing data to HeartDysfunctionReportCreated: " + e.__str__(), **log_args)
    pg_conn.dispose()
    
def create_consumer():
    creadentials =  read_credentials(  str(__location__), "rabbitmq")['credentials'] 
    connection = pika.BlockingConnection(pika.URLParameters(creadentials))
    channel = connection.channel()
    exchange_name = 'Events.Measurements:PolarH10DataUploaded'
    queue_name ='PolarH10DataUploaded'
    channel.exchange_declare( exchange= exchange_name ,  exchange_type= ExchangeType.fanout, durable = True)
    channel.queue_declare(queue=queue_name, durable = True)
    channel.queue_bind(queue_name, exchange_name)
    channel.basic_consume(queue=queue_name, auto_ack=True,
                          on_message_callback=on_request_message_received)
    channel.basic_qos(prefetch_count=1)
    print("Starting Server")
    channel.start_consuming()

create_consumer()

