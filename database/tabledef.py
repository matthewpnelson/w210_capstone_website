from sqlalchemy import *
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
 
engine = create_engine('sqlite:///alegorithm.db', echo=True)
Base = declarative_base()
 
########################################################################
class User(Base):
    """"""
    __tablename__ = "Users"
 
    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)
    name = Column(String)
    email = Column(String)
 
    #----------------------------------------------------------------------
    def __init__(self, username, password, name, email):
        """"""
        self.username = username
        self.password = password
        self.name = name
        self.email = email
		
class Beer(Base):
    """"""
    __tablename__ = "Beers"
 
    id = Column(Integer, primary_key=True)
    index = Column(String, unique=True)
    name = Column(String)
    description = Column(String)
    brewery = Column(String)
    style = Column(String)
    rating = Column(Float)
    raters = Column(Integer)
    monthly_stats = Column(Integer)
    unique_stats = Column(Integer)
    total_stats = Column(Integer)
    ibu = Column(Float)
    abv = Column(Float)
    label_url = Column(String)
    date_added = Column(DateTime)

	
    #----------------------------------------------------------------------
    def __init__(self, index, name, description, brewery, style, rating, raters, monthly_stats, unique_stats, total_stats, ibu, abv, label_url, date_added):
        """"""
        self.index = index
        self.name = name
        self.description = description
        self.brewery = brewery
        self.style = style
        self.rating = rating
        self.raters = raters
        self.monthly_stats = monthly_stats
        self.unique_stats = unique_stats
        self.total_stats = total_stats
        self.ibu = ibu
        self.abv = abv
        self.label_url = label_url
        self.date_added =date_added
 
# create tables
Base.metadata.create_all(engine)