import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data
import statsmodels.formula.api as sm


def assetPrice(stock):
    factors = pdr.DataReader(name='F-F_Research_Data_5_Factors_2x3_Daily',
                             data_source='famafrench')[0]
    print('Source data:')
    print(factors)
    factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
    factors['MKT'] = factors['MKT'] / 100
    factors['SMB'] = factors['SMB'] / 100
    factors['HML'] = factors['HML'] / 100
    factors['RMW'] = factors['RMW'] / 100
    factors['CMA'] = factors['CMA'] / 100
    stock_factor_mat = pd.merge(stock,
                                factors,
                                left_index=True,
                                right_index=True)  # Merging the stock and factor returns dataframes together
    stock_factor_mat['XsRet'] = (stock_factor_mat['Returns']
                                 - stock_factor_mat['RF'])  # Calculating excess returns

    CAPM = sm.ols(formula='XsRet ~ MKT',
                  data=stock_factor_mat).fit()
    print('CAPM R2:', CAPM.rsquared_adj)
    CAPM_beta = CAPM.params['MKT']
    print('CAPM beta:', CAPM_beta)
    
    FF3 = sm.ols(formula='XsRet ~ MKT + SMB + HML',
                 data=stock_factor_mat).fit()
    print('FF3 R2: ', FF3.rsquared_adj)

    FF5 = sm.ols(formula='XsRet ~ MKT + SMB + HML + RMW + CMA',
                 data=stock_factor_mat).fit()
    print('FF5 R2: ', FF5.rsquared_adj)

    CAPM_tstat = CAPM.tvalues
    FF3_tstat = FF3.tvalues
    FF5_tstat = FF5.tvalues

    print('CAPM p-value:', CAPM.pvalues)
    print('FF3 p-values:\n', FF3.pvalues)
    print('FF5 p-values:\n', FF5.pvalues)

    CAPM_coeff = CAPM.params
    FF3_coeff = FF3.params
    FF5_coeff = FF5.params

    results = pd.DataFrame({'CAPM_coeff': CAPM_coeff,
                            'CAPM_tstat': CAPM_tstat,
                            'FF3_coeff': FF3_coeff,
                            'FF3_tstat': FF3_tstat,
                            'FF5_coeff': FF5_coeff,
                            'FF5_tstat': FF5_tstat},
                           index=['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    return results


ticker = 'AAPL'

tickerFile = ticker + '.csv'
data = pd.read_csv(tickerFile, parse_dates=['Date'])
data = data.sort_values(by='Date')
data.set_index('Date', inplace=True)

data['Returns'] = data['Adj Close'].pct_change()  # Create daily returns column
returns = data['Returns'].dropna()  # Remove values of N/A

capm = assetPrice(data)
print(capm)
