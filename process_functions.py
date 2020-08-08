from datetime import datetime

def write_log(str):
    '''
    function used to write on the log.txt file the date and time and notes on the changes while updating the /input csv files
    '''
    # datetime object containing current date and time
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    log_file = open('log.txt', 'a')
    log_file.write(f'[{now}] {str}\n')
    log_file.close()

def applyDiff(row, df_province_second_to_last):
    row['daily'] = row['daily'] - int(df_province_second_to_last.loc[df_province_second_to_last['denominazione_provincia'] == row['denominazione_provincia'], 'totale_casi'])
    return row
