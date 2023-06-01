import pandas as pd
import numpy as np
import scipy as scipy
import statsmodels.api as sm
import statsmodels.tsa.x13 as X13
import pycountry
import seaborn as sb
import matplotlib.pyplot as plt

def mydateparser(dates):
    datestring = []
    for d in range(len(dates)):
        new = dates[d][:4] + 'Q' + dates[d][4:]
        datestring.append(new)
        # print(new)
    return datestring

def fixCols(df, total_columns):
    df.columns = [country.split(": ",1)[0] for country in df.columns]
    df = df.drop(columns = [col for col in df if col not in total_columns])
    date_list = df["date"].values.tolist()
    converted_dates = mydateparser([str(date) for date in date_list])
    df["date"] = pd.to_datetime(converted_dates)
    df = df.set_index("date")
    
    df = df.loc[:,~df.columns.duplicated()].copy()
    
    df = df.rename(columns={"U.S.": "United States", 
                            "U.K.": "United Kingdom", 
                            "Czech Republic": "Czechia", 
                            "South Korea": "Korea, Republic of", 
                            "Taiwan": "Taiwan, Province of China"})
    
    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_3

    df.columns = [countries.get(country, 'Unknown code') for country in df.columns]
        
    return df

def SeasonalAdjustment(df):
    for country in df:
        toSA = df[country]
        toSA = toSA[toSA.notnull()]
        
        # now lets seasonally adjust the data
        SAdata = X13.x13_arima_analysis(toSA,(2,1),maxdiff=(2,1),exog=None,log=None,outlier=True,trading=False,
                                        forecast_years=None,retspec=False,speconly=False,start=toSA.index[0],
                                        prefer_x13=True,freq='Q')
        # re-write data with seasonally adjusted data
        df.loc[:,country] = SAdata.seasadj

    return df

def LTDetrend(df):
    
    for country in df:
        to_detrend = df[country]

        # detrend the data
        LT = pd.Series(scipy.signal.detrend(to_detrend, axis=- 1, type='linear', bp=0, overwrite_data=False))
        LT = LT.set_axis(to_detrend.index.copy())
        
        # rewrite data with cyclical component
        df[country] = LT
    return df

def HPDetrend(df):
    
    for country in df:
        to_detrend = df[country]

        # detrend the data
        HP_cycle, HP_trend = sm.tsa.filters.hpfilter(to_detrend, 1600)

        # rewrite data with cyclical component
        # instead of rewriting the data, how do i save my output to a new dataframe (in the right corresponding location)?
        df[country] = HP_cycle
    return df

class Prepare_Correlations:
    def __init__(self, data, detrending, countries):
        # whatever measure of output is desired for use (we can use GDP, consumption industrial production, etc)
        self.data = data

        # which detrending technique you want to use (first difference, fourth difference, linear, HP)
        self.detrending = detrending

        # the list of countries to use
        self.countries = countries

    def tell_me_about_it(self):
        print("The detrending technique is: ", self.detrending)
        print("The included countries are: ", self.countries)
        print(self.data.head(10))

    def detrend(self, start_date, end_date):

        # restrict dates as specified
        self.data = self.data.loc[(self.data.index >= pd.to_datetime(start_date, format='%Y-%m-%d')) & (self.data.index <= pd.to_datetime(end_date, format='%Y-%m-%d'))]

        # restrict list of countries as specified
        self.data = self.data.drop(columns = [col for col in self.data if col not in self.countries], axis=1)

        # take natural log
        for country in self.data:
            self.data[country] = np.log(self.data[country])

        # Apply the selected detrending technique
        if self.detrending == 'HP Filter':
            # Apply HP filter detrending
            self.data = HPDetrend(self.data)

        elif self.detrending == 'linear detrending':
            # Apply linear detrending
            self.data = LTDetrend(self.data)

        elif self.detrending == 'first difference':
            # Apply log first differences detrending
            self.data = self.data.diff()

        else:
            raise ValueError("Invalid detrending technique. Choose 'HP Filter', 'linear detrending', or 'first difference'.")
        
        return self
    
    def get_correlationmatrix(self):
        correlationmatrix = self.data.corr().sort_values(by="USA", ascending=False)
        correlationmatrix = correlationmatrix[correlationmatrix.index]

        return correlationmatrix
    
    def get_heatmap(self):
        matrix = self.get_correlationmatrix()
        plt.figure(figsize=(35,20))
        # define the mask to set the values in the upper triangle to True
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        heatmap = sb.heatmap(matrix, mask=mask, vmin=-1, vmax=1, annot=False, cmap='RdYlGn')
        heatmap.set_title(self.detrending, pad=16);

    def get_organized(self):
        # import the correlation data (for now, use first differences)
        matrix = self.get_correlationmatrix()

        # reshape the data
        matrix_reshape = matrix.stack().reset_index()

        # rename the columns
        matrix_reshape.columns = ['iso3_firstcountry', 'iso3_secondcountry', 'correlation']

        # create mask that will remove duplicates
        mask_duplicates = (matrix_reshape[['iso3_firstcountry', 'iso3_secondcountry']].apply(frozenset, axis=1).duplicated()) | (matrix_reshape['iso3_firstcountry']==matrix_reshape['iso3_secondcountry'])
        allcorrelationdata = matrix_reshape[~mask_duplicates]

        return allcorrelationdata
