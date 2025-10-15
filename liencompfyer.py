from fyers_apiv3 import fyersModel
from datetime import date, timedelta

def responseData(redirect_uri,client_id,secret_key,token,sysm,sysm1,days,res, indexSym):

    grant_type = "authorization_code"                  ## The grant_type always has to be "authorization_code"
    response_type = "code"                             ## The response_type always has to be "code"
    state = "sample"                                   ##  The state field here acts as a session manager. you will be sent with the state field after successfull generation of auth_code
    #print(client_id,secret_key,redirect_uri)
    appSession = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
        state="sample_state"
    )
    auth_code=token
    appSession.set_token(auth_code)
    response = appSession.generate_token()

    try:
        access_token = response["access_token"]
        fyers = fyersModel.FyersModel(token=access_token,is_async=False,client_id=client_id,log_path="")
        current_date = date.today()
        formatted_date = current_date.strftime("%Y-%m-%d")
        ninety_days = timedelta(days=int(days))
        date_ninety_days_ago = current_date - ninety_days
        data = {"symbol":sysm,"resolution":f"{res}","date_format":"1","range_from":date_ninety_days_ago,"range_to":formatted_date,"cont_flag":"1"}
        sysmJson=fyers.history(data)
        #print("=======1=========")
        data = {"symbol":sysm1,"resolution":f"{res}","date_format":"1","range_from":date_ninety_days_ago,"range_to":formatted_date,"cont_flag":"1"}
        sysm1Json=fyers.history(data)
        #print("=======2=========")
        data = {
            "symbol": f"{indexSym}", # Symbol for the Nifty 50 Index on NSE
            "strikecount": 5
        }
        response = fyers.optionchain(data=data)
        #print("=======3=========")
        return [sysmJson,sysm1Json,response]
    except Exception as e:
       print(e,response)
       return [None,None,None]
    return [None,None,None]
