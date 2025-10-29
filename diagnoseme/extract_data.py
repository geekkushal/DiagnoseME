import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import pandas as pd  # Import Pandas for tabular data representation

# MongoDB connection setup
client = MongoClient()  # Replace with your MongoDB connection string if necessary
db = client['hospital_database']  # Create/use a database
collection = db['doctors']  # Create/use a collection
collection.delete_many({})

def kushal_extract_doctors():
    url = "https://www.kailashhealthcare.com/Specialities/physician"
    print("Trying to fetch data...")
    
    response = requests.get(url)
    if response.status_code == 200:  # Ensure the request was successful
        html_content = response.text
        
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'lxml')
        
        doctor_containers = soup.find_all('div', class_='ddoverlay')
        
        for container in doctor_containers:

            doctor_name = container.find('h3').text.strip() if container.find('h3') else "N/A"
            
            details = container.find_all('p')
            designation = details[0].text.strip() if len(details) > 0 else "N/A"
            area_of_expertise = (
                details[2].text.strip() if len(details) > 2 else "N/A"
            )  # 'Areas of Expertise'
            qualification = (
                details[4].text.strip() if len(details) > 4 else "N/A"
            )  # 'Qualification'
            
            doctor_data = {
                "name": doctor_name,
                "designation": designation,
                "area_of_expertise": area_of_expertise,
                "qualification": qualification,
            }
            
            collection.insert_one(doctor_data)
            print(f"Inserted: {doctor_data}")
        
        print("Data extraction and storage complete!")
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

def view_data():
    data = list(collection.find({}, {'_id': 0}))  # Exclude the MongoDB _id field for cleaner output

    df = pd.DataFrame(data)

    print("\nDoctors Data:")
    print(df)
    return df



#extract_doctors()
#df = view_data()
