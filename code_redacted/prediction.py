import numpy as np
import pandas as pd
from datetime import datetime as dt
import time

container_types = {
    # redacted
}

container_net_weights_full_pdf = {
    # redacted
}

container_net_weights_full_data = {
    # redacted
}

container_gross_weights_full_pdf = {
    # redacted
}

container_weights_empty_pdf = {
    # redacted
}

container_weights_empty_data= {
    # redacted 
}


def get_unknown_container_weights_dict(dict_in):

    dict = {}
    for ct, value in dict_in.items():
        if (np.isnan(value)):
            dict[ct] = value
    return dict

# calculate and print a dictionary of average weights for each container type
def get_mean_container_weights_dict(df):

    dict = {}
    df["net_weight_per_container"] = df.nettogewicht_in_kg / \
        df.angemeldete_containeranzahl

    for ct, value in dict.items():
        #print(ct)
        avg_weight =  df.query("angeforderter_behältertyp == @ct")["net_weight_per_container"].mean()
        dict[ct] = round(avg_weight, 2)

    return dict

def drop_single_pick_ups_and_single_initial_deliveries(df):
    """_summary_
     Use this function for a set of orders from one collection point before calculating prediction. 
     If you want to calculate accuracy after predicting use filter_for_min3_pick_ups_and_initial_deliveries instead because you need to split the latest orders off to to calculate accuracy.
    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # orders without single occurences of Abholauftrag and without Erstausstattung
    df1 = df.query("auftragstyp == 'Erstausstattung'")
    df_temp = df.query("auftragstyp != 'Erstausstattung'")
    # next line keeps data in df2 only if at least 2 rows after removing initial deliveries
    df2 = df_temp[df_temp.übergabestellennummer.duplicated(keep=False)]
    if df2.shape[0] > 1:
        # check if there are at least 2 orders with different pick-up date
        last_date = df2["abholdatum"].iloc[0]
        has_pick_up_range = False
        for i in range(df2.shape[0] - 1):
            date = df2["abholdatum"].iloc[i + 1]
            if date != last_date:
                has_pick_up_range = True
            last_date = date
        # without a time range between pick-ups we can not predict. 
        if not has_pick_up_range:
                return pd.DataFrame()
        concat_list = [df1, df2]
        df2 = pd.concat(concat_list)
        df2 = df2.sort_values(by=["abholdatum", "auftragsnummer"])
    return df2

'''
pseudo code: 
- sort table first by collection_point and then by collection_date ascending
- **for each collection_point**
    - *(1) calculate how many kg a collection point collects each day*
    - 
    - total_days_of_collection_all_orders = latest_collection_date - earliest_collection_date
    - total_collected_net_weight_kg_all_orders = sum(net_weight_all_orders without the earliest order)
    - *daily_collection_kg* = total_collected_net_weight_kg_all_orders / total_days_of_collection_all_orders
        - TODO OPTIONAL: daily_collection_kg = <other-calculated-weight> / total_days_of_collection_all_order
    - 
    - *(2) calculate container capacity in kg for one collection point*
    - 
    - for each container_type: 
        - containers_type_in_stock = all_delivered_containers - all_picked_up_containers
        - capacity_kg_container_type = containers_type_in_stock * container_net_weights_full_pdf
        - save capacity_kg_container_type in list for collection_point
        - save containers_type_in_stock in list for collection_point
    - *capacity_kg_collection_point* = sum(list of capacity_kg_container_type)
    - 
    - *(3) calculate how many days to fill containers in stock*
    - 
    - *days_to_fill_capacity_collection_point* = capacity_kg_collection_point / daily_collection_kg
    - 
    - *(4) calculate the day when collection point's capacity is full*
    - 
    - **day_collection_point_is_full** = latest_collection_date_collection_point + days_to_fill_capacity_collection_point
'''
def predict_capacity_of_collection_point_full_date(df, c_weight_dict=container_net_weights_full_pdf, debug=False):

    dict = predict_base(df, c_weight_dict, debug)

    return dict


def predict_base(df, c_weight_dict, debug):
    
    dict = {
        'timestamp' : pd.Timestamp(dt.now()),
        'übergabestellennummer' : df["übergabestellennummer"].iloc[0],
        'konzernnummer': df["konzernnummer"].iloc[0],
        'name_1' : df["name_1"].iloc[0],
        'volle_Addresse' : df["volle_Addresse"].iloc[0],
        'lat' : df["lat"].iloc[0],
        'long' : df["long"].iloc[0],
        'typ': df["typ"].iloc[0], 
        'vertragsnummer': df["vertragsnummer"].iloc[0],
        'transporteur' : df["transporteur"].iloc[0],
    }

    if(debug):
        print(f"calculationg status for collection point: {df.übergabestellennummer.iloc[0]}")

    # (1) calculate how many kg a collection point collects each day
    latest_collection_date = df["abholdatum"].iloc[-1]
    earliest_collection_date = df["abholdatum"].iloc[0]
    total_days_of_collection_all_orders = latest_collection_date - earliest_collection_date
    total_collected_net_weight_kg_all_orders = df["nettogewicht_in_kg"].iloc[1:].sum()

    daily_collection_kg = total_collected_net_weight_kg_all_orders / total_days_of_collection_all_orders.days
    if(debug):
        print(f"(1) calculate how many kg a collection point collects each day")
        print(f"latest_collection_date: {latest_collection_date}")
        print(f"earliest_collection_date: {earliest_collection_date}")
        print(f"total_days_of_collection_all_orders: {total_days_of_collection_all_orders}")
        print(f"total_collected_net_weight_kg_all_orders: {total_collected_net_weight_kg_all_orders}")
        print(f"daily_collection_kg: {daily_collection_kg}")

    # (2) calculate container capacity in kg for one collection point
    if(debug):
        print()
        print(f"(2) calculate container capacity in kg for one collection point")
    container_types = list(df.gelieferter_behältertyp.unique())
    capacity_kg_collection_point = 0
    container_type_count = 0
    for ctype in container_types:
        container_type_count += 1
        all_delivered_containers = df.query("gelieferter_behältertyp == @ctype")["gelieferte_behälteranzahl"].iloc[:].sum()
        all_picked_up_containers = df.query("angeforderter_behältertyp == @ctype and auftragsstatus == 'Erledigt'")["angemeldete_containeranzahl"].iloc[1:].sum()
        containers_type_in_stock = all_delivered_containers - all_picked_up_containers
        capacity_kg_container_type = containers_type_in_stock * c_weight_dict[ctype]
        capacity_kg_collection_point += capacity_kg_container_type.astype(int)

        dict["container_typ_" + str(container_type_count)] = ctype
        dict["container_typ_" + str(container_type_count) + "_bestand"] = containers_type_in_stock 

        if(debug):
            print(f"container_type_count {container_type_count} of type {ctype}")
            print(f"all_delivered_containers for type {ctype}: {all_delivered_containers}")
            print(f"all_picked_up_containers for type {ctype}: {all_picked_up_containers}")
            print(f"containers_type_in_stock for type {ctype}: {containers_type_in_stock}")
            print()

    if(debug):
        print(f"capacity_kg_collection_point: {capacity_kg_collection_point} kg")
    
    # (3) calculate how many days to fill containers in stock
    if(debug):
        print()
        print(f"(3) calculate how many days to fill containers in stock")
    days_to_fill_capacity_collection_point = capacity_kg_collection_point / daily_collection_kg
    if(debug):
        print(f"days_to_fill_capacity_collection_point: {days_to_fill_capacity_collection_point} days")

    # (4) calculate the day when collection point's capacity is full
    latest_collection_date = latest_collection_date.date()
    days_to_fill_capacity_collection_point = str(days_to_fill_capacity_collection_point) + " days"
    days_to_fill_capacity_collection_point = pd.Timedelta(days_to_fill_capacity_collection_point)
    day_collection_point_is_full = latest_collection_date + days_to_fill_capacity_collection_point
    
    
    if(debug):
        print()
        print(f"(4) calculate the day when collection point's capacity is full")
        print(type(latest_collection_date))
        print(type(days_to_fill_capacity_collection_point))
        print()
        print(f"day_collection_point_is_full: {day_collection_point_is_full}")              

    latest_collection_date = df["abholdatum"].iloc[-1]
    days_since_latest_collection = pd.Timestamp(dt.now()) - latest_collection_date

    weight_collected_today = days_since_latest_collection.days * daily_collection_kg

    capacity_currently_filled = 0
    day_collection_point_is_full_timestamp = None
    # when there are no containers in stock, nothing can be collected so capacity_currently_filled is 0.
    if capacity_kg_collection_point > 0:
        capacity_currently_filled = round(weight_collected_today / capacity_kg_collection_point * 100, 2)
        day_collection_point_is_full_timestamp = pd.Timestamp(day_collection_point_is_full)
       
    dict["kapa_kg"] = capacity_kg_collection_point
    dict["tägl_smenge_kg"] = round(daily_collection_kg, 2)
    dict["letzte_abholung_datum"] = latest_collection_date
    dict["tage_seit_abholung"] = days_since_latest_collection.days
    dict["erreicht_kg"] = round(weight_collected_today, 2)
    dict["erreicht_prozent"] = capacity_currently_filled
    dict["kapa_erreicht_tage"] = days_to_fill_capacity_collection_point.days
    dict["kapa_erreicht_am"] = day_collection_point_is_full_timestamp
    
    if(debug):
        print(f"returning dict collection point: {df.übergabestellennummer.iloc[0]}")
        print(dict)
    return dict


# this function predicts based on a df filtered on 1 company_group and 1 container_type
# TODO: convert to other variable names
def predict_collection_point_full(df_konzern_container_type, mean_weight):

    total_netto_weight_in_kg = df_konzern_container_type["nettogewicht_in_kg"].iloc[1:].sum()  

    #total_container_angemeldet = df_konzern_container_type["angemeldete_containeranzahl"].iloc[1:].sum()
    #total_container_geliefert = df_konzern_container_type["gelieferte_behälteranzahl"].iloc[1:].sum()

    total_time = df_konzern_container_type["abholdatum"].iloc[-1] - df_konzern_container_type["abholdatum"].iloc[0]
    durchschnitt_tag = total_netto_weight_in_kg/total_time.days
    prediction_voll = mean_weight/durchschnitt_tag
    prediction_sammelstelle_voll_in_Tagen = prediction_voll*df_konzern_container_type["gelieferte_behälteranzahl"].iloc[-1]
    prediction_sammelstelle_voll_in_Tagen_str = prediction_sammelstelle_voll_in_Tagen.astype(str) + " days"
    prediction_sammelstelle_voll_in_Tagen_td = pd.Timedelta(prediction_sammelstelle_voll_in_Tagen_str)
    letzte_abholung = df_konzern_container_type["abholdatum"].iloc[-1]
    

    pred_sammelst_voll_datum = letzte_abholung + prediction_sammelstelle_voll_in_Tagen_td
    return pred_sammelst_voll_datum



def filter_dataframe_for_prediction(orders_, debug=True):
    """_summary_ 
    filters out non plausible data and data that we technically can not use for predictions

    Parameters
    ----------
    orders_ : _type_
        _description_
    debug : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_ 
    completed orders and uncompleted as separate dataframes
    """ 

    # drop pick-up orders with wrong data: completed withoud pick-up date and without weight
    q_completed_pick_up_orders_missing_data = "auftragsstatus == 'Erledigt' and auftragstyp == 'Abholauftrag' and abholdatum == '1999-01-01' and nettogewicht_in_kg == 0"
    print(f"drop q_completed_pick_up_orders_missing_data rows: {orders_.query(q_completed_pick_up_orders_missing_data).shape[0]}")
    orders_ = orders_.drop(orders_.query(q_completed_pick_up_orders_missing_data).index)

    # if "angemeldete containerzahl" means the amount of containers that was actually picked up...we could calculate the nettogewicht_in_kg. 
    # if it's just the amount of empty containers they want to get delivered ..we can not use thosw rows for prediction.
    # drop them for now and ask TODO
    q_completed_pick_up_orders_missing_weight = "auftragsstatus == 'Erledigt' and nettogewicht_in_kg == 0 and auftragstyp == 'Abholauftrag'"
    print(f"drop q_completed_pick_up_orders_missing_data rows: {orders_.query(q_completed_pick_up_orders_missing_weight).shape[0]}")
    orders_ = orders_.drop(orders_.query(q_completed_pick_up_orders_missing_weight).index)

    q_completed_initial_deliveries_missing_delivery_date = "auftragsstatus == 'Erledigt' and abholdatum == '1999-01-01' and auftragstyp == 'Erstausstattung'"
    print(f"initial delivery without a delivery date looks like they never delivered. we drop them: {orders_.query(q_completed_initial_deliveries_missing_delivery_date).shape[0]} rows")
    orders_ = orders_.drop(orders_.query(q_completed_initial_deliveries_missing_delivery_date).index)

    if(debug):
        print(orders_.shape[0])

    # filter open orders
    q_open_orders = "abholdatum == '1999-01-01'"
    orders_open = orders_.query(q_open_orders)
    orders_comp = orders_.drop(orders_.query(q_open_orders).index)

    print(f"after dropping because they are not plausible ...")
    print(f"we keep {orders_open.shape[0]} open orders")
    print(f"and keep {orders_comp.shape[0]} completed orders")
    return orders_comp, orders_open


def filter_for_report(orders_, debug=True):
    """_summary_
    Filters out data that should not be contained in the reporting layer.
    Parameters
    ----------
    orders_ : _type_
        _description_
    debug : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_ filtered dataframe
    """
    # filter certain collection points based on stakeholders insights: 103508246
    q_übergabestellenummern = "übergabestellennummer == 103508246"
    print(f"drop q_übergabestellenummern rows: {orders_.query(q_übergabestellenummern).shape[0]}")
    orders_ = orders_.drop(orders_.query(q_übergabestellenummern).index)
    return orders_ 

def remove_orders_with_unknown_weights(df, use_pdf_weight=True, debug=False):

    if use_pdf_weight:
        weight_dict = container_net_weights_full_pdf
    else:
        weight_dict = container_net_weights_full_data

    drop_list = list(get_unknown_container_weights_dict(weight_dict).keys())
    if debug:
        print(f"unknown container weights: {drop_list}")
    df = df[df.gelieferter_behältertyp.isin(drop_list) == False]
    df = df[df.angemeldete_containeranzahl.isin(drop_list) == False]
    return df


def filter_for_min3_pick_ups_and_initial_deliveries(df):
    """_summary_
    Use this function for a set of orders from one collection point before calculating prediction
    IF you want to split the df in train and test set for later accuracy calculation.

    If you just want to predict you need only 2 orders and can use function
    drop_single_pick_ups_and_single_initial_deliveries instead.

    This funciton filters the df so that at least 3 pick-up orders are contained
    and the latest order date is not an initial delivery!

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    Returns a df that is filtered as described. The df will be empty if the above conditions don't match!
    (e.g. the given df contains only 2 pick-up orders.)
    """
    # orders without single occurences of Abholauftrag and without Erstausstattung
    df1 = df.query("auftragstyp == 'Erstausstattung'")
    df2 = df.query("auftragstyp != 'Erstausstattung'")
    #df2 = df_temp[df_temp.übergabestellennummer.duplicated(keep=False)] # we need at least 3!
    
    if df2.shape[0] > 2:
        concat_list = [df1, df2]
        df2 = pd.concat(concat_list)
        df2 = df2.sort_values(by=["abholdatum", "auftragsnummer"])
        while df2.iloc[-1].auftragstyp == 'Erstausstattung':
            df2 = df2.iloc[:-1]
        return df2
    return pd.DataFrame()

def calculate_error(x_train, y, c_weights=container_net_weights_full_pdf, debug=False):
    """_summary_ 

    Parameters
    ----------
    x_train : _type_
        _description_
    container_net_weights_full_pdf : _type_
        _description_
    debug : bool, optional
        _description_, by default False

    Returns
    -------
    _type_ int
        _description_ the error in days as int
    """
    pred_dict = predict_capacity_of_collection_point_full_date(x_train, container_net_weights_full_pdf, debug)
    y_pred_1 = pred_dict["kapa_erreicht_am"] # pd.Timestamp
  
    if debug == True:
        cp_no = pred_dict["übergabestellennummer"]
        print(f"calculate_error: y_pred_1 type = {type(y_pred_1)} value is {y_pred_1} for cp {cp_no}")

    if y_pred_1 is None:
        raise ValueError("error can not be calculated with None value prediction.")
    else:
        error = y - y_pred_1
        return np.abs(error).days


    """_summary_
    The training data includes all orders except the latest order date which is used as test data.
    Returns:
    _type_: pandas dataframe
        _description_ the training data
    _type_: pandas Timestamp
        _description_ the latest collection date    
    """
def train_test_split(df, debug=False):
    latest_collection_date = df["abholdatum"].iloc[-1]
    x_train = df.query("abholdatum != @latest_collection_date")
    y = df.query("abholdatum == @latest_collection_date")["abholdatum"].iloc[0].date()
    if debug == True:
        print(f"y date = {y}")
    y = pd.Timestamp(y)
    if debug == True:
        print(f"y Timestamp = {y}")

    return x_train, y

def get_accuracy(error_list, threshold_miss_hit):
    """_summary_
    We define a **MISS** prediction as being more than x  away from the real value. 
    In our case the real value is the month of the real pick-up date.
    A **HIT** is a prediction that has the same month as the real value.

    The accuracy is the proportion of HITs of all predictions.
    Parameters
    ----------
    error_list : _type_
        _description_
    threshold_miss_hit : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    ###
    hit = 0
    for e in error_list:
        if e[1] < threshold_miss_hit:
            hit += 1

    accuracy = hit / len(error_list)
    return round(accuracy, 2)

def get_mean_error(error_list):
    esum = 0
    for e in error_list:
        #print(esum)   
        esum += e[1]

    mean_error = esum / len(error_list)
    return round(mean_error, 2)


def print_prediction_metrics(error_list, hit_threshold=30):
    print(f"Predicting all collection_points results in ")
    print(f"Mean Error of: {get_mean_error(error_list)} days ")
    print(f"Accuracy of: {get_accuracy(error_list, hit_threshold) * 100}% (using a threshold of {hit_threshold} days)")
    print()
    print(pd.DataFrame(error_list).describe().round().T)


