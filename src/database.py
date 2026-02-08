from pymongo import MongoClient

uri = "mongodb+srv://Try:radhey@gymkaro.pqsy3gu.mongodb.net/?appName=GymKaro"
client = MongoClient(uri)

# Database (created on first write)
db = client["User_Info"]
# collection = db.Exercise_Info
# # Insert data (THIS creates the DB)
# result = collection.insert_one({
#     "name": "Rudra",
#     "email": "rudra@example.com"
# })
# fetch = collection.find_one({'email':'rudra@example.com'})
# # print("Database and collection created!")
# print(fetch['name'])
# print(type(fetch))

# personal_attributes = ['name', 'age', 'gender', 'height', 'weight']
db.Personal_Info.delete_many({})
def new_user(email, attributes):
    
    if (db.Personal_Info.count_documents({'email':email}) != 0):
        print("User Already Exists")
        return False
    db.Personal_Info.insert_one({'email':email, **attributes})
    db.Exercise_Info.insert_one({'email':email})
    return True

attributes = {'name':'Rudra', 'age':19}
new_user('test@123', attributes)

def fetch_personal_info(email, *args)->dict|None|list:
    result = db.Personal_Info.find_one({'email':email})
    if (result is None):
        print('Not able to Fetch')
        return None
    if not args:
        return result
    return [result.get(value) for value in args]

def add_exercise_info(email, **kwargs):
    find = 0
    change = 0
    if 'summary' in kwargs:
        result = db.Exercise_Info.update_one({'email':email}, {'$set':{'summary':kwargs.get('summary')}})
        find = find | result.matched_count
        change = change | result.modified_count
    if 'exercise_id' in kwargs:
        exercise_id = kwargs.get('exercise_id')
        result = db.Exercise_Info.update_one({'email':email}, {"$set":
            {f'exercise.{exercise_id}':kwargs.get('info')}
        })
        find = find | result.matched_count
        change = change | result.modified_count

    if (find):
        print('User Find')
    if (change):
        print('User Details Updated')
# exercises = {'exercise_id':1}
add_exercise_info('test@123', exercise_id=1, info={'mistake':'Glt hai bsdk', 'feedback':'Mat Kar Bsdk'})
    

