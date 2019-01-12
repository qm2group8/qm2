
# coding: utf-8

# In[526]:


#Begin with importing a variety of libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import seaborn as sns
import cufflinks as cf
import plotly.plotly as py
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import shapely

get_ipython().run_line_magic('matplotlib', 'inline')


# In[527]:


cf.go_offline() #Offline use for Plotly/Cufflinks
init_notebook_mode(connected=True)


# # USA County Data
# 
# We'll start by gathering demographic data about US Counties. The US Department of Agriculture provides some comprehensive datasets about US counties: https://www.ers.usda.gov/data-products/atlas-of-rural-and-small-town-america/download-the-data/. There are four relevant datasets, so we'll import then merge them at the county codes (FIPS).

# In[528]:


#Read in people data
peopledata = pd.read_csv('Rural_Atlas_Update18/People.csv')

# retain relevant columns
peopledata = peopledata[['FIPS','State','County','WhiteNonHispanicPct2010','BlackNonHispanicPct2010',
                         'AsianNonHispanicPct2010','NativeAmericanNonHispanicPct2010',
                         'HispanicPct2010','ForeignBornPct','ForeignBornEuropePct', 
                         'ForeignBornMexPct', 'NonEnglishHHPct','Ed1LessThanHSPct',
                         'Ed5CollegePlusPct', 'AvgHHSize', 'FemaleHHPct']]

# We'll take a quick look at the dataframe

peopledata.head(2)


# In[529]:


# read in jobs data

jobsdata = pd.read_csv('Rural_Atlas_Update18/Jobs.csv')

# Retain the relevant columns

jobsdata = jobsdata[['FIPS','UnempRate2017','PctEmpAgriculture','PctEmpMining','PctEmpConstruction','PctEmpManufacturing','PctEmpGovt']]

jobsdata.head()


# In[530]:


# read in income/finance data

incomedata = pd.read_csv('Rural_Atlas_Update18/Income.csv')

# Retain the columns of interest
incomedata = incomedata[['FIPS','PovertyAllAgesPct','PerCapitaInc']]

incomedata.head(1)


# In[531]:


classification.head()


# In[532]:


# read in the county classification data
# This .csv has a different encoding, so we'll change it from the UTF-8 default to ISO-8859-1
classification = pd.read_csv('Rural_Atlas_Update18/County Classifications.csv',encoding="ISO-8859-1")

# retain relevant columns

classification = classification[['FIPStxt','Low_Education_2015_update','Hipov_1115']]

# Rename columns
classification = classification.rename({'FIPStxt':'FIPS','Low_Education_2015_update':'low_edu',
                                        'Hipov_1115':'high_poverty'}, axis=1)

classification.head(1)


# In[533]:


# Merge all four together on the "FIPS" column,
# which are unique county codes. 
county_data = peopledata.merge(jobsdata,on='FIPS').merge(incomedata,on='FIPS').merge(classification,on='FIPS')
county_data.head(2)


# Next, we'll bring in data about crime rates. An important note to make here is that these rates represent reported incidents of crime, rather than actual crime rates – which, in some cases, can vary significantly. This data is retrieved from the FBI: https://ucr.fbi.gov/crime-in-the-u.s/2016/crime-in-the-u.s.-2016/cius-2016.

# In[534]:


# read in crime data, eliminate commas with the "thousands" command
crime = pd.read_csv('CountyOffenses.csv',thousands=',')


# In[535]:


# Let's preview the crime data
crime.head()


# In[536]:


# There was an indexing error when trying to 
# access columns. We'll see what the issue is by
# looking at the column names. 
crime.columns 


# In[537]:


# It appears to contain line separators '\n' 
# so we'll remove them from the column names.

crime = crime.rename({'Violent\ncrime':'Violent Crime',
                      'Murder and\nnonnegligent\nmanslaughter':'Murder and Nonnegligent Manslaughter',
                     'RapeRevised':'Rape (Revised)',
                      'RapeLegacy':'Rape (Legacy)',
                     'Aggravated assault':'Aggravated Assault',
                      'Property\ncrime':'Property Crime',
                      'Larceny-\ntheft':'Larceny-Theft',
                     'Motor\nvehicle theft':'Motor Vehicle Theft'}, axis=1)


# In[538]:


# Check column names again
crime.columns


# In[539]:


# We'll keep only the relevant columns.
# The "Rape" columns were removed because of the 
# two different definitions. It significantly reduces the 
# sample size (half of the data is linked to each definition) and adversely 
# affects correlations. 

crime = crime[['State','County','Violent Crime','Murder and Nonnegligent Manslaughter',
               'Robbery','Aggravated Assault','Property Crime','Burglary']]


# In[540]:


# Merge the county data 
uscounties = pd.merge(county_data,crime,on=['County','State'],how='left')


# Coming up, we'll add in water and air contamination data. 
# 
# Water data will be sourced from the Environmental Protection Agency: https://catalog.data.gov/dataset/safe-drinking-water-information-system-sdwis-federal-reports-advanced-search-tool. This dataset shows the mean value of drinking water standard violations from 1994-2016.
# 
# Air quality data also comes from the EPA: https://www.epa.gov/air-trends/air-quality-cities-and-counties.

# In[541]:


watervio = pd.read_csv('mean_water_violations_1994_2016.csv')


# In[542]:


watervio.head(1)


# In[543]:


# We only have a limited sample, although it will suffice for our purposes.
# About 1/3 of counties will be represented in our analysis. 
# This should be noted as a potential limitation.
len(watervio)


# In[544]:


watervio = watervio.drop(['FIPS2','CNTY_FIPS','State','State_County','STATE_FIPS','COUNTY'],axis=1)
watervio = watervio.rename({'mean_viol_cnty':'mean water violations'},axis=1)
watervio.head()


# In[545]:


# merge water violation data
uscounties = uscounties.merge(watervio,on='FIPS',how='left')


# In[546]:


uscounties.head()


# In[547]:


# Data matched for 1084 counties.
len(uscounties[uscounties['mean water violations']>0])


# Now we'll introduce the air quality data:

# In[548]:


# import air quality data
air = pd.read_csv('airquality_county_2017.csv')


# In[549]:


air.head()


# In[550]:


# ND Stands for not disclosed. We'll replace them with NaN.
cols = ['2010 Population', 'CO 8-hr', 'PB 3-mo',
       'NO2 AM', 'NO2 1-hr', '03 8-hr', 'PM10 24-hr', 'PM2.5 Wtd AM',
       'PM2.5 24-hr', 'SO2 1-hr']
air[cols] = air[cols].replace({'ND':np.nan})
air[cols] = air[cols].replace({'IN':np.nan})


# In[551]:


# Drop State and County columns
air = air.drop(['State','County'],axis=1)


# In[552]:


# We have data for 1146 counties. Again this represents about a third of all counties.
# Again, this should be noted as a constraint.
len(air) 


# In[553]:


# Convert data to floats

air[['2010 Population', 'CO 8-hr', 'PB 3-mo',
       'NO2 AM', 'NO2 1-hr', '03 8-hr', 'PM10 24-hr', 'PM2.5 Wtd AM',
       'PM2.5 24-hr', 'SO2 1-hr']] = air[['2010 Population', 'CO 8-hr', 'PB 3-mo',
       'NO2 AM', 'NO2 1-hr', '03 8-hr', 'PM10 24-hr', 'PM2.5 Wtd AM',
       'PM2.5 24-hr', 'SO2 1-hr']].astype(float)


# In[554]:


len(air)


# In[555]:


uscounties = uscounties.merge(air,on='FIPS',how='left')


# In[556]:


len(uscounties[uscounties['SO2 1-hr']>0])


# Moving forward, we'll now add in cancer mortality rates. This will be particularly interesting because it can show the serious impact that environmental contaminants can have. The data here will represent mortality rates for the year 2014. This data is retrieved from the Institute for Health Metrics: http://ghdx.healthdata.org/organizations/institute-health-metrics-and-evaluation-ihme.

# In[557]:


cancer_all = pd.read_csv('county_cancer_mortality_2014.csv')


# In[558]:


cancer_all.head(2)


# In[559]:


# It looks like there are confidence intervals included in the data.
# We'll clean the data to only grab the first value, rather than the interval. 

# First, we'll create a function that splits a string at the spaces
# then takes the first element of the split string - in this case, the value we're looking for. 

def splitrates(x):
    try:
         return float(x.split()[0])
    except: 
        pass

# Next we'll apply this to every cell in the DataFrame. 

cancer = cancer_all.applymap(splitrates)


# In[560]:


cancer['FIPS'] = cancer_all['FIPS']
cancer['Location'] = cancer_all['Location']


# In[561]:


# Drop USA Value, as it is NaN and prevents conversion of FIPS to integers
cancer = cancer.drop(0)
cancer.head(1)


# In[562]:


#Take only rows with a valid FIPS value
cancer = cancer[np.isfinite(cancer['FIPS'])]


# In[563]:


#Convert all FIPS to integers so they will merge with the rest of the dataset.
cancer.FIPS = [int(i) for i in cancer.FIPS]


# In[564]:


uscounties = uscounties.merge(cancer,on='FIPS',how='left')


# Theres one more dataset about cancer we'll use from the CDC: https://www.cdc.gov/cancer/uscs/dataviz/download_data.htm, https://gis.cdc.gov/Cancer/USCS/DataViz.html. It describes cancer rates per 100,000 by county, 1994-2014.

# In[565]:


cdc = pd.read_csv('USCS_1999_2015_ASCII/BYAREA_COUNTY.txt',sep='|')


# In[566]:


# We'll filter the columns to only include cancer incident rates,
# all kinds of cancer, for both male and females; and data for all races. 
cdc = cdc[(cdc['SITE']=='All Cancer Sites Combined') & (cdc['EVENT_TYPE']=='Incidence') & (cdc['SEX']=='Male and Female') & (cdc['RACE']=='All Races')]
cdc.head()


# In[567]:


# Grab FIPS values that are between the parentheses in 'AREA' column
cdc['FIPS'] = [int(i.split('(')[1].split(')')[0]) for i in cdc.AREA]


# In[568]:


#Rename rows
cdc = cdc[['AGE_ADJUSTED_RATE','FIPS']].rename({'AGE_ADJUSTED_RATE':'cancer_rate'},axis=1)


# In[569]:


# The rates are input as strings and the dataset contains "~" and "." for unavailable data.
# We'll convert them to floats and 'coerce' the string values to NaN

cdc['cancer_rate'] = pd.to_numeric(cdc['cancer_rate'],errors='coerce')


# In[570]:


cdc.head()


# In[571]:


uscounties = uscounties.merge(cdc,on=['FIPS'],how='left')


# Next, we'll read in TRI Release data from: https://www.epa.gov/trinationalanalysis/supporting-data-files-2016-tri-national-analysis. This data is about toxic waste release in counties in 2016. An often referenced factor in environmental justice is the release of these toxins and chemicals. 

# In[572]:


# There are commas in the values so we'll 
# use the 'thousands' parameter to remove them.
tri = pd.read_csv('county_release_data.csv',thousands=',')


# In[573]:


tri.head(1)


# In[574]:


# We'll remove redundant columns, then join it with the US Counties dataframe
tri = tri.drop(['County Name','State Abbreviation','Population (2010)'],axis=1)


# In[575]:


# Append to uscounties
uscounties = uscounties.merge(tri,on=['FIPS'],how='left')


# To finish up, we'll add in data that represents the diversity of a county using the Simpson index, found on Kaggle here: https://www.kaggle.com/mikejohnsonjr/us-counties-diversity-index/home.

# In[576]:


# Import diversity index data
diversityindex = pd.read_csv('countydiversityindex.csv')

# retain relevant columns
diversityindex = diversityindex[['Location','Diversity-Index']]

diversityindex.head()


# In[577]:


#This data has location and state in the same columns, as well as no FIPS data. 
#To merge it, we should try to isolate county names, then merge on both county name and state (as many counties share names) )

#Looking at the data, the county name should appear before the comma in the location column.
#We can split the name at the column, then add everything before the comma to a new column.
diversityindex[['County','State']] = diversityindex.Location.str.split(',', expand=True)


diversityindex.head(3)


# In[578]:


#The counties are listed only by name. 
#We need to remove all mentions of "County","City",and "Borough" from the diversity index.

terms = ['City','Borough','County','Census','Area']   
diversityindex['County'] = diversityindex.County.str.replace('County', '') #There are errors adding the list in.
diversityindex['County'] = diversityindex.County.str.replace('City', '')   #Each term will be done separately.
diversityindex['County'] = diversityindex.County.str.replace('Census', '')
diversityindex['County'] = diversityindex.County.str.replace('Area', '')
diversityindex['County'] = diversityindex.County.str.replace('Borough', '')
diversityindex['County'] = diversityindex.County.str.replace('Parish', '')
diversityindex.head()


# In[579]:


#Drop redundant column
diversityindex = diversityindex.drop(['Location'],axis=1)


# In[580]:


#There are some states written in uppercase.
#Counties follow the "title" format, so we can remove the states by filtering by ".istitle".

diversityindex = diversityindex[diversityindex.County.str.istitle()]
diversityindex = diversityindex.astype(float,errors='ignore')
diversityindex.head(3)


# In[581]:


#For some reason, the data is not merging. 
county_data_test = pd.merge(uscounties,diversityindex,on=['County','State'])
county_data_test.head()


# In[582]:


#To problemshoot, we'll check the datatypes. 
type(diversityindex['County'][10]), type(uscounties['County'][10]) 


# In[583]:


#Both appear to be strings, so there should not be an issue.
# We'll check corresponding lengths to see if there are any issues

print(len(diversityindex['County'][10]), len(uscounties['County'][10]),
      "\n",len(diversityindex['State'][10]), len(county_data['State'][10]))   
#The length is different, there may be extra spaces.


# In[584]:


# We'll try removing double spaces

diversityindex.County = diversityindex.County.str.replace('  ','')      
diversityindex.State = diversityindex.State.str.replace('  ','')
uscounties_test_2 = pd.merge(county_data,diversityindex,on='County')
len(uscounties_test_2) 


# In[585]:


# Now instead of 0, we have matched 12
# Identifying the length issue may have been essential. Removing a single space at the end of a word may be difficult.
# If we use the same procedure as the double-space, we will lose the space inbetween multiple-word strings.
# Luckily, there is a specialized command:
# .rstrip() removes whitespace at the end of a string.

diversityindex.County = diversityindex.County.str.rstrip() 
diversityindex.State = diversityindex.State.str.rstrip()
diversityindex.State = diversityindex.State.str.replace(' ','')


# In[586]:


# Now we can try to merge the datasets again:
uscounties = pd.merge(uscounties,diversityindex,on=['County','State'],how='left')

# We're merging using how='left' just in case the Kaggle dataset is incomplete
uscounties.head(10)


# # USA Shapefiles
# Next, we'll build a Geodataframe. This allows for the choropleth plots of our data. We'll use shapefiles of the US counties and merge them with our uscounties dataframe. The shapefiles are available from the US census Bureau: https://www.census.gov/geo/maps-data/data/cbf/cbf_counties.html

# In[587]:


# read in GeoPandas shapefile data
countyshapes = gpd.read_file('UScb_county_shapefile/cb_2017_us_county_20m.shp')
countyshapes['FIPS'] = countyshapes['GEOID'].astype(int)
countyshapes.head(2)


# In[588]:


#We'll merge the datasets on FIPS and see if this works for choropleths with shapefiles.
uscounties= countyshapes.merge(uscounties,on='FIPS')
uscounties.head(2)      #To plot using shapefile data, it must be in the GeoPandas Dataframe format.
                            #This is why we've created a new dataframe with a new name. 


# In[589]:


# Remove erroneous commas that appeared in the Burglary column

uscounties['Burglary'] = uscounties['Burglary'].replace(',','')


# In[590]:


# Export as a csv
uscounties.to_csv('county_data_111.csv')


# # Sample Choropleth of US Counties
# 
# We now have the uscounties dataframe, which is a comprehensive collection of statistics about US counties. Let's make a sample choropleth.
# 

# In[592]:


# This variable is easy to change and is what will be included on the map

argument = 'HispanicPct2010' 
                             # It is called in .plot() as the column argument

vmin, vmax = 0, 1            # Similarly, these arguments are the max/min values for the legend. 
                             # Again, easy to change if we change the variable.
    
fig, ax = plt.subplots(1, figsize=(25, 12))

#Alaska and Hawaii make the formatting of the map awkward, so they are omitted from the visualization. 
#They can be visualized separately, if necessary.

uscounties[(uscounties['State']!='AK')&(uscounties['State']!='HI')].plot(column=argument, 
                                            cmap='plasma', linewidth=1.0, ax=ax, edgecolor='0.8')
ax.axis('off')

#Title
ax.set_title('Hispanic or Latino Proportion of County Population (2010)',               fontdict={'fontsize': '20',
                        'fontweight' : '5'})

#Source Annotation
ax.annotate('US Census Bureau, 2017',
           xy=(0.1, .08), xycoords='figure fraction',
           horizontalalignment='left', verticalalignment='top',
           fontsize=10, color='#555555')

#Colorbar Legend
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)

# There will be an error resulting from some null values.
# This is due to incomplete data in the Kaggle diversity dataset.
# NaN values are plotted as 0. Although there are some clear outliers from this, general trends are still observable.


# We can use Plotly for an interactive choropleth. Plotly takes in the FIPS data and builds a map around it:

# In[593]:


#plotly.tools.set_credentials_file(username='chasemcd', api_key='eE22MdAcTKR4rztgcYXt') #If Plotly isn't working offline, remove '#' 
                                                                                        #Ive added my account so the plots will work.
fips_data = uscounties['FIPS']
values = uscounties['UnempRate2017'].round(2) #Here we are mapping the percentage of county votes for the GOP in the 2016 election
layout = go.Layout(showlegend=False)

    
fig = ff.create_choropleth(fips=fips_data, values = values,
                           round_legend_values=True,legend_title='Unemployment Rate',
                           title='Unemployment Rate by County (2017)')
iplot(fig,show_link=False)

#There will be several errors, but we can ignore them. 
#The first is a sort error with the current version of Pyplot.
#The second is an error that references state and country level FIPS codes, which throws an error because we are plotting by county.


# # Analysis
# We'll now move into our the data analysis portion of the project. 
# This will include statistical correlations as well as exploration through visualization.

# In[ ]:


# Let's get a dataframe of the correlations between our variables
# This will be quite a large table, and is simply exploratory. 
# It can act as a springboard for further analysis. 

corrtable = uscounties.corr(method='spearman').abs().unstack()
unstack = corrtable.sort_values(kind="quicksort", ascending=False)
corr_df = pd.DataFrame(unstack)
corr_df.head(1) #Unable to rename column '0' or index it. 


# In[ ]:


#Export and Reimport as CSV to rename column '0' and variable colums

corr_df.to_csv('Corr Dataframe - Column Names Missing.csv')


# In[ ]:


correlations = pd.read_csv('Corr Dataframe.csv')
correlations.head(1)


# In[ ]:


#Elimite rows where var1 and var2 are equal
correlations = correlations[correlations['var1']!=correlations['var2']]

# The 10 highest correlations with median income:
correlations[correlations['var1']=='Median Income'].head(10)


# In[ ]:


# The 10 highest correlations with PM2.5 air contamination:
correlations[correlations['var1']=='PM2.5 24-hr'].head(10)


# In[ ]:


# Now we'll look at statistically significant correlations: That is, the Spearman correlation with p-values.
# We'll use variables that represent demographic information and environmental justice factors. 

# First, import the stats package:

from scipy.stats.stats import spearmanr


# In[ ]:


# Remove identifying columns – that is, those that
# aren't variables we'll be calculating correlations from
cols = [x for x in list(uscounties.columns) if x not in ['STATEFP','COUNTYFP','COUNTYNS','AFFGEOID',
                                        'GEOID','NAME','LSAD','ALAND','AWATER',
                                        'geometry','FIPS','State','County']]


# In[ ]:


# We'll split the the columns into two lists: 
# one that represents demographic information, 
# another that represents environmental factors

all_env = [ 'mean water violations','CO 8-hr','PB 3-mo','NO2 AM','NO2 1-hr',
       '03 8-hr','PM10 24-hr','PM2.5 Wtd AM','PM2.5 24-hr','SO2 1-hr','Cervical Cancer Mortality',
       'Breast Cancer Mortality','Uterine Cancer Mortality','Ovarian Cancer Mortality',
       'Prostate Cancer Mortality','Testicular Cancer Mortality','Kidney Cancer Mortality',
       'Brain/Nervous System Cancer Mortality','Mesothelioma Mortality ','Hodgkin Lymphoma Mortality',
       'Leukemia Mortality','Multiple Myeloma Mortality','Thyroid Cancer Mortality','Bladder Cancer Mortality',
       'Trachael, Bronchus, and Lung Cancer Mortality','Larynx Cancer Mortality','Pancreatic Cancer Mortality',
       'Gallbladder and Biliary Tract Cancer','Liver Cancer Mortality','Colon and Rectum Cancer Mortality',
       'Stomach Cancer Mortality','Esophageal Cancer Mortality','Lip and Oral Cavity Cancer Mortality',
       'Neoplasms Mortality','cancer_rate','Location','Count of Total Facilities','Total Gross Releases (lb)',
        'Gross Releases per Sq Mile (lb)','Rank Based on Gross Releases per Sq Mile',
        'Count of Facilities reporting Water Releases','Water Releases (lb)','Water Releases per Sq Mile (lb)',
        'Rank Based on Water Releases per Sq Mile','Count of Facilities reporting Air Releases','Air Releases (lb)',
           'Air Releases per Sq Mile (lb)','Rank Based on Air Releases per Sq Mile',
           'Count of Facilities Reporting Land Releases','Land Releases (lb)','Land Releases per Sq Mile (lb)',
           'Rank Based on Land Releases per Sq Mile','RSEI Pounds','RSEI Hazard','RSEI Score',
           'Rank Based on RSEI Pounds','Rank Based on RSEI Hazard','Rank Based on RSEI Score']

demo = [x for x in cols if x not in all_env]

env = [ 'mean water violations','PB 3-mo',
       '03 8-hr','PM2.5 24-hr','cancer_rate','Breast Cancer Mortality',
       'Trachael, Bronchus, and Lung Cancer Mortality','Colon and Rectum Cancer Mortality','Air Releases (lb)',
      'RSEI Hazard','RSEI Score','RSEI Pounds','Land Releases (lb)','Water Releases (lb)','Total Gross Releases (lb)',
      'Count of Total Facilities']


# In[ ]:


# create an empty DataFrame to store the data in

corr_data = {'variable_1':[],'variable_2':[],'spearman_correlation':[],'p-value':[]}
spearman = pd.DataFrame(data=corr_data)
spearman.reset_index()


# In[ ]:


# Create all empty cells

for i in range(len(env)*len(demo)):
    spearman.loc[i] = np.nan


# In[ ]:


# Iterate through all the variables and append the relevant information.

count = 0
for i in demo:
    for x in env:
        corr, p_value = spearmanr(uscounties[i],uscounties[x],nan_policy='omit')
        spearman['variable_1'][count] = i
        spearman['variable_2'][count] = x
        spearman['spearman_correlation'][count] = corr
        spearman['p-value'][count] = float(p_value)
        count += 1


# In[ ]:


# Take only statistically significant correlations (p<0.05)
spearman = spearman[spearman['p-value']<0.05]

# Sort by correlation
spearman = spearman.sort_values('spearman_correlation',ascending=False)


# In[ ]:


# Now we'll take a look at the correlations

spearman[0:10]


# In[ ]:





# In[ ]:


sns.heatmap(uscounties[['NonEnglishHHPct','UnempRate2017','PctEmpManufacturing','Gini Index','Diversity-Index',]].corr().abs())


# In[ ]:


sns.lmplot(x='PM2.5 24-hr',y='HispanicPct2010',data=uscounties)


# In[ ]:


sns.lmplot(x='Diversity-Index',y='mean water violations',data=uscounties)


# In[ ]:


uscounties[['mean water violations','Diversity-Index','Gini Index','PM2.5 24-hr']].corr(method='spearman')


# In[ ]:


uscounties.to_csv('uscounties test.csv')


# In[ ]:


sns.lmplot(x='PM2.5 24-hr',y='ForeignBornMexPct',data=uscounties)


# In[ ]:


sns.lmplot(x='Median Income',y='cancer_rate',data=uscounties)


# In[ ]:


sns.lmplot(x='PM2.5 24-hr',y='HispanicPct2010',data=uscounties)


# Now we'll move into actual statistical tests using the Scipy statistics package. 
# 
# We'll compare deviations from the mean in population groups. That is, if communities whose minority population exceeds the average is statistically different from communities whose White Non-Hispanic population is greater than or equal to the average. 

# In[ ]:


data = [go.Heatmap(z=uscounties.values.tolist(), colorscale='Viridis')]

py.iplot(data)


# In[ ]:


# Import statistical package "stats" from SciPy
from scipy import stats


# In[ ]:


# First, calculate the mean value for each group. 
white_mean = uscounties['WhiteNonHispanicPct2010'].mean()
hispanic_mean = uscounties['HispanicPct2010'].mean()
black_mean = uscounties['BlackNonHispanicPct2010'].mean()
asian_mean = uscounties['AsianNonHispanicPct2010'].mean()
na_mean = uscounties['NativeAmericanNonHispanicPct2010'].mean()
mixed_mean = uscounties['MultipleRacePct2010'].mean()


# In[ ]:


# Let's try testing above mean hispanic population air pollution to above mean white population air pollution.
white_above_pm25 = uscounties[uscounties['WhiteNonHispanicPct2010'] >= white_mean]['PM2.5 24-hr'].dropna()
hispanic_above_pm25 = uscounties[uscounties['HispanicPct2010'] >= hispanic_mean]['PM2.5 24-hr'].dropna()
stats.ttest_ind(white_above_pm25, hispanic_above_pm25)


# In[ ]:


# Black
black_above_pm25 = uscounties[uscounties['BlackNonHispanicPct2010'] > black_mean]['PM2.5 24-hr'].dropna()
stats.ttest_ind(white_above_pm25, black_above_pm25)


# In[ ]:


# Asian
asian_above_pm25 = uscounties[uscounties['AsianNonHispanicPct2010'] > asian_mean]['PM2.5 24-hr'].dropna()
stats.ttest_ind(white_above_pm25, asian_above_pm25)


# In[ ]:


# Native American
na_above_pm25 = uscounties[uscounties['NativeAmericanNonHispanicPct2010'] > na_mean ]['PM2.5 24-hr'].dropna()
stats.ttest_ind(white_above_pm25, na_above_pm25)


# In[ ]:


# Mixed Race
mixed_above_pm25 = uscounties[uscounties['MultipleRacePct2010'] > mixed_mean]['PM2.5 24-hr'].dropna()
stats.ttest_ind(white_above_pm25, mixed_above_pm25)


# In[ ]:


# There is a statistically significant difference in PM2.5 air contamination in communities
# that have above-mean proportions of minority populations for all groups except Asians. 

# Let's visualise the average PM2.5 contamination levels in these communities.

#Create a new dataframe with only the necessary columns
# pm_df = uscounties[['PM2.5 24-hr','MultipleRacePct2010','NativeAmericanNonHispanicPct2010','AsianNonHispanicPct2010',
                #'BlackNonHispanicPct2010','HispanicPct2010','WhiteNonHispanicPct2010']]

#pm_df = pd.DataFrame()
    
#pm_df['Mixed'] = uscounties[uscounties['MultipleRacePct2010'] >= mixed_mean]['PM2.5 24-hr']
#pm_df['Native American'] = uscounties[uscounties['NativeAmericanNonHispanicPct2010'] >= na_mean ]['PM2.5 24-hr']
#pm_df['Black'] = uscounties[uscounties['BlackNonHispanicPct2010'] >= black_mean]['PM2.5 24-hr']
#pm_df['White'] = uscounties[uscounties['WhiteNonHispanicPct2010'] >= white_mean]['PM2.5 24-hr']
#pm_df['Hispanic'] = uscounties[uscounties['HispanicPct2010'] >= hispanic_mean]['PM2.5 24-hr']

#pm_df.iplot(kind='box')


# In[ ]:


#pm_df.mean().iplot(kind='bar')


# In[ ]:


from scipy.stats.stats import pearsonr



# from scipy.stats.stats import pearsonr
# 

# Next, we'll add in US voting data for the 2016 presidential election. It inidicates the amount and proportion of votes for democrats/republicans.
# We can find election data here: https://github.com/tonmcg/US_County_Level_Election_Results_08-16.
# I'll download this and add it to the running dataframe.

# In[ ]:


#electiondata = pd.read_csv('US_County_Level_Election_Results_08-16-master/2016_US_County_Level_Presidential_Results.csv')
#electiondata.head(1) #Again we see there is county name and state abbreviation, we'll drop these then combine at FIPS


# In[ ]:


#electiondata = electiondata.drop(['state_abbr','county_name','diff','per_point_diff','Unnamed: 0','votes_dem','votes_gop','total_votes'],axis=1)
#electiondata[['per_dem','per_gop']] = electiondata[['per_dem','per_gop']].astype(float,errors='ignore')
#electiondata.head(1)


# In[ ]:


#electiondata = electiondata.rename(index=str, columns={'combined_fips':'FIPS',
                                                      #'per_dem':'Percent Dem 2016',
                                                      #'per_gop':'Percent GOP 2016',
                                                      #})
#electiondata.head(1)


# In[ ]:


#uscounties = county_data.merge(electiondata,on='FIPS')
#uscounties.head(1)


# In[ ]:


#water = pd.read_csv('water_system_summary.csv')


# In[ ]:


#water = water.rename({'State Code':'State','Counties Served':'County'},axis=1)
#water = water.drop(['Unnamed: 4','Unnamed: 5','Unnamed: 6'],axis=1)
#water.head(1)


# In[ ]:


# Drop commas, normalize county names.
#water['PopulationServed Count'] = water['PopulationServed Count'].replace(',', '',regex=True)
#water['PopulationServed Count'] = water['PopulationServed Count'].astype(float)

#water['PopulationServed Count'] = water['PopulationServed Count'].replace(',', '',regex=True)
#water['Number of Violations'] = water['Number of Violations'].astype(float)


#water['County'] = water.County.str.replace('County', '')
#water['County'] = water.County.str.replace('City', '')
#water['County'] = water.County.str.replace('Census', '')
#water['County'] = water.County.str.replace('Census-Area', '')
#water['County'] = water.County.str.replace('Area', '')
#water['County'] = water.County.str.replace('Borough', '')
#water['County'] = water.County.str.replace('Parish', '')
#water['County'] = water.County.str.replace('Municipality', '')

#water.head(1)


# In[ ]:


#len(water) # There are roughly 146,000 rows for just over 3,100 counties. 


# In[ ]:


# Aggregate information about counties together
# Start with just a dataframe of State Codes and Counties

#agg_water = uscounties[['State','County']]
#agg_water['Population Served'] = 0
#agg_water['Water Std Violations'] = 0
#agg_water.head()


# In[ ]:


#for i, row in water.iterrows():
#    for x, rows in agg_water.iterrows():
#        if (agg_water['State'][x] == water['State'][i]) and (agg_water['County'][x]==water['County'][i]):
#            agg_water['Population Served'][x] = agg_water['Population Served'][x] + water['PopulationServed Count'][i]
#            agg_water['Water Std Violations'][x] = agg_water['Water Std Violations'][x] + water['Number of Violations'][i]            


# In[ ]:


#water['Population Total'] = water.groupby(['State', 'County'])['PopulationServed Count'].transform('sum')
#water['Violations Total'] = water.groupby(['State', 'County'])['Number of Violations'].transform('sum')
#water = water.drop_duplicates(subset=['State', 'County'])


# In[ ]:


#water = water.drop(['PopulationServed Count','Number of Violations'],axis=1)


# In[ ]:


#We'll create a measure of violations as a proportion of served population, then drop the population column.
#water['waterviolation_prop'] = water['Violations Total']/water['Population Total']
#water.head()


# In[ ]:


#water = water.drop(['Population Total'],axis=1)


# In[ ]:


#len(water.County)


# In[ ]:


#uscounties = uscounties.merge(water,on=['State','County'],how='left')


# In[ ]:


#uscounties.head()


# We'll also bring in normalized GINI index data from GitHub: https://github.com/iloveluce/mini-graph. The data is from the US Government, but a comprehensive version of the original dataset couldn't be located, so we've found an altered version that has normalized rates. The GINI score represents a measure of income inequality, with 0 being the most equal and, in this case of a normalized value, 1 being the most unequal.

# In[ ]:


# import GINI index data 
#gini = pd.read_json('gini_data.json')
#gini.head(1)


# In[ ]:


# The 'id' column represents 'FIPS'. We'll change the name and drop all other columns.
#gini['FIPS']=gini['id']
#gini=gini.drop(['Gini Margin of Error','id','name'],axis=1)
#gini['Gini Index'] = gini['Gini Index'].astype(float)
#gini.head(1)


# Moving on, we'll add in data on average Medicare reimbursements per person. This data might be useful in analyzing medical costs associated with higher environmental contamination. Data retrieved from: https://datausa.io/map/?level=county&key=total_reimbursements_b, originally produced by the US Government. 

# In[ ]:


# Read in medicare data
#medicare = pd.read_csv('Medicare Reimbursements.csv')


# In[ ]:


#medicare.head(1)


# In[ ]:


# Take the most recent data
#medicare = medicare[['geo_name','total_reimbursements_b_2014']]

#medicare.head(1)


# In[ ]:


# Split geo_name column to 'State' and 'County'
#medicare[['County', 'State']] = medicare.geo_name.str.split(', ', expand = True)


# In[ ]:


# Rename, convert to float, and drop unnecessary columns
#medicare['medicare reimbursements (2014)'] = medicare['total_reimbursements_b_2014'].astype(float)
#medicare = medicare.drop(['geo_name','total_reimbursements_b_2014'],axis=1)


# In[ ]:


#Clean up County column for merging

#medicare['County'] = medicare.County.str.replace('County', '')
#medicare['County'] = medicare.County.str.replace('City', '')   
#medicare['County'] = medicare.County.str.replace('Census', '')
#medicare['County'] = medicare.County.str.replace('Census-Area', '')
#medicare['County'] = medicare.County.str.replace('Area', '')
#medicare['County'] = medicare.County.str.replace('Borough', '')
#medicare['County'] = medicare.County.str.replace('Parish', '')
#medicare['County'] = medicare.County.str.replace('Municipality', '')

# .rstrip() removes whitespace at the end of a string
#medicare.County = medicare.County.str.rstrip() 
#medicare.State = medicare.State.str.rstrip()


# In[ ]:


#medicare.head(1)


# In[ ]:


# Since there's no FIPS data,
# we'll merge it on the County and State columns with the larger dataset.

#uscounties= pd.merge(county_data,medicare,on=['County','State'],how='left')




# In[ ]:


#uscounties.head(1)

