import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tabledef import *
 
engine = create_engine('sqlite:///alegorithm.db', echo=True)
 
# create a Session
Session = sessionmaker(bind=engine)
session = Session()
 
user = User("msaedi","password", 'Mehdi Saedi', 'msaedi@berkeley.edu')
session.add(user)
 
user = User("fakbar","1234", 'Faeze Akbar', 'faezeh.akbar@gmail.com')
session.add(user)

user = User("fasedi","1234", 'Farokh Saedi', 'farokh.saedi@gmail.com')
session.add(user)

user = User("mdanesh","1234", 'Mahshid Danesh', 'mahshid.danesh@me.com')
session.add(user)
 
# commit the record the database
session.commit()
 
session.commit()