import os 
from datetime import datetime

def get_model_and_date_from_files():
    cwd = os.getcwd()

    file_names = [f for f in os.listdir(cwd) if f.endswith('_Log.txt')]

    model_names = []
    date_times  = []
     

    # Model_SDP_Graph_JustTheta2024-09-30_Monday_17-35_Log.txt

    for file_name in file_names:
        file_name    = file_name[:-8] # Drop _Log.txt
        time_of_day  = file_name[-5:]
        file_name    = file_name[:-6] # Drop _time
        day_of_week  = file_name.split('_')[-1]
        file_name    = file_name[:-len(day_of_week)-1] # Drop _day
        date         = file_name[-10:]
        file_name    = file_name[:-10]
        # print(file_name)

        model_name   = file_name[6:] # Drop Model_
        
        model_names.append(model_name)
        date_times.append((date,day_of_week,time_of_day))

    return model_names,date_times

def sort_by_date_time(model_names,date_times):
    # Sort by date time
    date_time_strs = [date + '_' + day + '_' + time for date,day,time in date_times]
    date_time_strs = [datetime.strptime(date_time_str,'%Y-%m-%d_%A_%H-%M') for date_time_str in date_time_strs]
    model_names,date_times = zip(*sorted(zip(model_names,date_times),key=lambda x: x[1]))
    # reformat date_time_strs back from datetime objects
    date_times = [(date,day,time) for date,day,time in date_times]   
    return model_names,date_times

def print_all_logs():

    model_names,date_times = get_model_and_date_from_files()
    model_names,date_times = sort_by_date_time(model_names,date_times)
    # Longest model name
    max_model_name_len = max([len(model_name) for model_name in model_names])

    # longest day of the week is wednesday duh
    max_day_len = 9

    # Print the logs
    print("Model Name".ljust(max_model_name_len+2) + "Date".ljust(10) + "Day".ljust(max_day_len) + "Time")
    print("-"*(max_model_name_len+2+10+max_day_len+5))
    for i in range(len(model_names)):
        model_name = model_names[i]
        date_time  = date_times[i]

        date = date_time[0]
        day  = date_time[1]
        time = date_time[2]

        print(model_name.ljust(max_model_name_len+2) + date.ljust(10) + ' ' + day.ljust(max_day_len)  + ' ' + time)

if __name__ == "__main__":

    print_all_logs()
    
    # cwd = os.getcwd()
    # model_names , date_times = get_model_and_date_from_files()
    # print(model_names)
    # print(date_times)
        

    
        