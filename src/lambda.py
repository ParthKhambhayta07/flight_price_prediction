import json
from io import StringIO
import io
import boto3
import pandas as pd
import numpy as np

email_client = boto3.client('sns')
s3 = boto3.client('s3')
runtime= boto3.client('runtime.sagemaker')


def data_pre_processing(input_df):
    input_df["Journey_day"] = pd.to_datetime(input_df.Date_of_Journey, format="%d/%m/%Y").dt.day.astype('int')
    input_df["Journey_month"] = pd.to_datetime(input_df.Date_of_Journey, format = "%d/%m/%Y").dt.month.astype('int')
    
    # dropping "Date_of_Journey" column, since it has no use
    input_df.drop(["Date_of_Journey"], axis = 1, inplace = True)
    input_df["Dep_hour"] = pd.to_datetime(input_df.Dep_Time).dt.hour.astype('int')
    input_df["Dep_min"] = pd.to_datetime(input_df.Dep_Time).dt.minute.astype('int')
    
    # dropping "Dep_Time" column, since it has no use
    input_df.drop(["Dep_Time"], axis = 1, inplace = True)
    
    input_df["Arrival_hour"] = pd.to_datetime(input_df.Arrival_Time).dt.hour.astype('int')
    input_df["Arrival_min"] = pd.to_datetime(input_df.Arrival_Time).dt.minute.astype('int')
    
    # dropping "Arrival_Time" column, since it has no use
    input_df.drop(["Arrival_Time"], axis = 1, inplace = True)
    
    hour_into_minute = pd.to_numeric(input_df['Duration'].str.replace(r'\D+', ' ', regex=True).str.split(' ').str[0])*60
    mins = pd.to_numeric(input_df['Duration'].str.replace(r'\D+', ' ', regex=True).str.split(' ').str[1])
    
    # some rows dont have mins. hence will become NaN if add hr and min columns tgt. 
    input_df['Duration_min'] = np.where(mins.isnull(), hour_into_minute, hour_into_minute + mins )
    input_df.drop(["Duration"], axis = 1, inplace = True)
    
    # converting Airline categorical value into nominal
    airline_cols = ['Airline_Air Asia','Airline_Air India','Airline_GoAir','Airline_IndiGo','Airline_Jet Airways','Airline_Jet Airways Business','Airline_Multiple carriers','Airline_Multiple carriers Premium economy','Airline_SpiceJet','Airline_Trujet','Airline_Vistara','Airline_Vistara Premium economy']
    airline_df = pd.DataFrame(columns=airline_cols)
    for i in range(len(input_df)):
        col_name = [col for col in airline_df.columns if str(input_df.loc[i]["Airline"]) in col]
        cols = { str(col_name[0]):1 }
        airline_df.loc[i] = cols
    
    airline_df = airline_df.fillna(0)

    # converting Source categorical value into nominal
    source_cols = ['Source_Banglore','Source_Chennai','Source_Delhi','Source_Kolkata','Source_Mumbai']
    source_df = pd.DataFrame(columns=source_cols)

    for i in range(len(input_df)):
        col_name = [col for col in source_df.columns if str(input_df.loc[i]["Source"]) in col]
        cols = { str(col_name[0]):1 }
        source_df.loc[i] = cols

    source_df = source_df.fillna(0)

    # converting Destination categorical value into nominal
    destination_cols = ['Destination_Banglore','Destination_Cochin','Destination_Delhi','Destination_Hyderabad','Destination_Kolkata','Destination_New Delhi']
    destination_df = pd.DataFrame(columns=destination_cols)

    for i in range(len(input_df)):
        col_name = [col for col in destination_df.columns if str(input_df.loc[i]["Destination"]) in col]
        cols = { str(col_name[0]):1 }
        destination_df.loc[i] = cols
    destination_df = destination_df.fillna(0)

    # converting additional_info categorical value into nominal
    additional_info_cols = ['Additional_Info_1 Long layover','Additional_Info_1 Short layover','Additional_Info_2 Long layover','Additional_Info_Business class','Additional_Info_Change airports','Additional_Info_In-flight meal not included','Additional_Info_No Info','Additional_Info_No check-in baggage included','Additional_Info_No info','Additional_Info_Red-eye flight']
    additional_info_df = pd.DataFrame(columns=additional_info_cols)
    
    for i in range(len(input_df)):
        col_name = [col for col in additional_info_df.columns if str(input_df.loc[i]["Additional_Info"]) in col]
        cols = { str(col_name[0]):1 }
        additional_info_df.loc[i] = cols

    additional_info_df = additional_info_df.fillna(0)


    input_df.drop(["Route"], axis = 1, inplace = True)
    input_df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
    input_df = pd.concat([input_df, airline_df, source_df, destination_df, additional_info_df], axis = 1)
    input_df.drop(["Airline", "Source", "Destination", "Additional_Info"], axis = 1, inplace = True)
    return input_df
    
def predict_flight_price(input_features, endpoint_name):
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='text/csv',
                                       Body=input_features)
    
    result = json.loads(response['Body'].read().decode())
    return result

def lambda_handler(event, context):
    
    records = [x for x in event.get('Records', []) if x.get('eventName') == 'ObjectCreated:Put']
    sorted_events = sorted(records, key=lambda e: e.get('eventTime'))
    latest_event = sorted_events[-1] if sorted_events else {}
    info = latest_event.get('s3', {})
    input_key = info.get('object', {}).get('key')
    bucket_name = info.get('bucket', {}).get('name')
    
    print("dynamic S3 params")
    print(input_key)
    print(bucket_name)
    
    # bucket_name = 'mlsb-batch7-group-b'
    # input_key = 'endpoint_testing/endpoint_testing.csv'
    output_key = 'model_prediction_output/predicted_result.csv'
    endpoint_name = 'xgboost-2023-09-19-08-47-21-291'
    
    topic_arn = 'arn:aws:sns:us-east-1:853513120488:mlsb-batch-7-group-b-sns'
    
    # reading test data from s3 location
    response = s3.get_object(Bucket=bucket_name, Key=input_key)
    data = response['Body'].read()
    input_df = pd.read_csv(io.BytesIO(data))
    df=input_df.copy()

    # pre-process test data
    pre_processed_df = data_pre_processing(input_df)
    predicted_flight_price_df = pd.DataFrame(columns=['predicted_flight_price'])
    
    # for each sample record predicting the flight price
    for index, row in pre_processed_df.iterrows():
        input_features = ','.join(str(value) for value in row)
        print(input_features)
        predicted_flight_price = predict_flight_price(input_features, endpoint_name)
        predicted_flight_price_df.loc[index] = { 'predicted_flight_price': predicted_flight_price }
        
    predicted_flight_price = pd.concat([df, predicted_flight_price_df], axis = 1)

    predicted_flight_price_csv = StringIO()
    predicted_flight_price.to_csv(predicted_flight_price_csv, index=False)
    
    # uploading predicted result at s3 location
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=predicted_flight_price_csv.getvalue() )
    
    # sending sns alert
    email_client.publish(TopicArn=topic_arn, Message='Please find the predicted flight price at s3 location : s3://mlsb-batch7-group-b/model_prediction_output/predicted_result.csv')
    
    return {
        'statusCode': 200,
        'body': json.dumps('Flight price prediction Lambda execution completed...')
    }
