import pandas as pd
import pickle
import json
import numpy as np
import string
import re


# For handling the typeError: Object of type int64 is not JSON serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Load the preprocessed pickle Dataset
dict_df = pickle.load(file=open(file="Model And New dataset/updatedDf.pkl", mode="rb"))

# Make dict to pandas DataFrame
df = pd.DataFrame(data=dict_df)

# Load the model
model = pickle.load(file=open(file="Model And New dataset/CV_similarity.pkl", mode="rb"))


# Our product recommended function
def get_productid_categoryid(product_name: str):
    global category_id
    product_id_list = []
    product_info = {
        'product id': [],
        'category id': 0
    }
    try:
        product_name = product_name.lower()

        product_category_id = int(df[df['name'] == product_name]['product_category_id'].values[0])

        category_id = product_category_id

        product_index = df[df['name'] == product_name].index[0]
        distances = sorted(list(enumerate(model[product_index])), reverse=True, key=lambda x: x[1])

        for i in distances[1:]:
            pid = df.loc[i[0], 'id']
            product_category_id2 = int(df.loc[i[0], 'product_category_id'])
            if product_category_id == product_category_id2:
                product_id_list.append(pid)
            else:
                pass

    except:
        return False

    else:
        try:
            for i in product_id_list:
                product_info['product id'].append(i)
                product_info['category id'] = category_id
        except:
            return False
        else:
            json_data = json.dumps(obj=product_info, cls=NpEncoder)
            return json_data


if __name__ == "__main__":
    productName = input("Enter your product name: ")
    response = get_productid_categoryid(product_name=productName)

    if response:
        print("Recommended product: ", response)
        # print(type(response))
    else:
        print("Opps!! Something is wrong!!")
