import pandas as pd
import glob

if __name__ == '__main__':
    path = './Data'
    all_files = glob.glob(path + r"/*.TXT")
    li = []  # 建立一個空的 list
    for filename in all_files:
        # iterate through each file in the "data" folder that ends with ".TXT"
        df = pd.read_csv(filename, names=[
            'State', 'Gender', 'Year', 'Name', 'No. of Occurrences'])
        li.append(df)

    # concat all the dataframes
    all_df = pd.concat(li, axis=0, ignore_index=True)

    # write out the data to a csv file
    all_df.to_csv(path + '/all_states.csv', index=False)
