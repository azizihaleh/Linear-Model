from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


style.use('fivethirtyeight')
data_1 = pd.read_csv("C:\\Users\Haleh Azizi\Desktop\job\data_1_3.cvs.csv")

X = data_1['x']
Y = data_1['y']

sns.pairplot(data_1) # Distribution od Dataset
plt.show()

sns.heatmap(data_1.corr(),annot=True) # Correlation Between Variables
plt.show()

def squer_error(Y_orig,Y_line):   # Defining the Loss Function
    return sum((Y_line - Y_orig)**2)   # Sum of Squared Error (Loss Function)


def r_squer(Y_orig,Y_line): # Defining R Squared Value
    Y_mean_line = [mean(Y_orig)]
    squered_error_regr = squer_error(Y_orig,Y_line)
    squered_error_mean = squer_error(Y_orig,Y_mean_line)
    return 1 - (squered_error_regr/squered_error_mean)
    

def a_b(x,y):                # Calculation (Optimization) of a (Slope) and b (Intercept)
    a = (((mean(x)*mean(y)) - mean(x*y))/
         ((mean(x)**2)-mean(x**2))) # Slope

    b = mean(y) - a* mean(x) # Intercept

    return a , b


a , b = a_b(X,Y)
linear_reg = [(a*x)+b for x in X]

r = r_squer(Y,linear_reg)
print('\n\n',r)
SSE = squer_error(Y,linear_reg)
print('\n\n',SSE)

plt.scatter(X,Y,c = 'b') # Plotting Dataset
plt.plot(X,linear_reg) # Plotting the Linnear Regression
plt.title('Linear Model using OLS Approach',fontsize=15)
plt.xlabel(print('Y = {a} X + {b}' .format (a = a, b = b)))
plt.xlabel('X',fontsize=12)
plt.ylabel('Y',fontsize=12)
plt.show()

plt.scatter(Y,linear_reg) # Scatter Plot Between Y and Predicted Y
plt.title('Scatter Plot Between Y and Predicted Y',fontsize=15)
plt.xlabel('Y',fontsize=12)
plt.ylabel('Predicted Y',fontsize=12)
plt.show()


# Calculation of Residuals, Noise which is Conditioned to X
residuals = list(np.array(Y) - np.array(linear_reg))
sns.distplot((residuals))
plt.title('PDF of Error of LM',fontsize=15)
plt.xlabel('Residuals',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.show()


# Second Degree Polynomial Modeling of Residuals, Noises which are Conditioned to X
degree = 2
sqrresiduals = list((np.array(Y) - np.array(linear_reg))**2)
weights = np.polyfit(X, sqrresiduals, degree) # Weights of 2nd Degree P.M.
#weights = [4.91,4.56,4.73]
model = [(weights[0]*x**2+weights[1]*x+weights[2]) for x in X]

plt.xlabel(print('Squared Residuals = {a} X ^ 2 + {b} X + {c}' .format (a = weights[0], b = weights[1], c = weights[2])))
plt.scatter(X,residuals)  # Plotting the Residuals
plt.plot(X,model) # Plotting the P.M on Residuals
plt.title('Polynomial Model on Squared Residuals',fontsize=15)
plt.xlabel('X',fontsize=12)
plt.ylabel('Squared Residuals',fontsize=12)
plt.show()


# Defining a New Linear Regression Considering Polynomial Model over Residuals
# Detecting and Resolving Heteroskedasticity

# Defining the Matrix of X = (1,X)
matrix_x = np.zeros((len(X),2)) 
x_1 = np.ones((len(X),1))
x_2 = np.array(X)
matrix_x[:,:1] = x_1
matrix_x[:,1] = x_2
matrix_x


# Defining the Matrix of weights
matrix_w = np.zeros((len(X),len(X))) 
weight_s = [1/k for k in model]

def replaceDiagonal(matrix, replacementList):
    for i in range(len(replacementList)):
        matrix[i][i] = replacementList[i]
        
replaceDiagonal(matrix_w,weight_s)


# Calculation of intercept and slope using the equation B=(inv(X'WX))X'WY
# B_1 = inv(X'WX)    and    B_2 = X'WY
B_1 = np.linalg.inv(np.matmul(np.matmul(matrix_x.transpose(), matrix_w), matrix_x))
B_2 = np.matmul(np.matmul(matrix_x.transpose(), matrix_w), Y)
B = np.matmul(B_1, B_2)


linear_reg_final = [(B[1]*x) + B[0] for x in X]

r = r_squer(Y,linear_reg_final)
print('\n\n',r)
SSE = squer_error(Y,linear_reg_final)
print('\n\n',SSE)

plt.scatter(X,Y,c = 'b') # Plotting Dataset
plt.plot(X,linear_reg_final) # Plotting the Modified Linnear Regression
plt.title('Linear Model using WLS Approach',fontsize=15)
plt.xlabel(print('Y = {a} X + {b}' .format (a = B[1], b = B[0])))
plt.xlabel('X',fontsize=12)
plt.ylabel('Y',fontsize=12)
plt.show()



# Defining a New Linear Regression Considering Polynomial Model over Residuals
# Detecting and Resolving Heteroskedasticity
#weight = [1/k for k in model]
weight = [1/(k**0.5) for k in model]
Xnew = [m*n for m,n in zip(X,weight)]
Ynew = [m*n for m,n in zip(Y,weight)]
aopt , bopt = a_b(np.array(Xnew),np.array(Ynew))
linear_reg_new = [(aopt*x)+bopt for x in X]

r = r_squer(Y,linear_reg_new)
print('\n\n',r)
SSE = squer_error(Y,linear_reg_new)
print('\n\n',SSE)

plt.scatter(X,Y,c = 'b') # Plotting Dataset
plt.plot(X,linear_reg_new) # Plotting the Modified Linnear Regression
plt.title('Linear Model using Semi-WLS Approach',fontsize=15)
plt.xlabel(print('Y = {a} X + {b}' .format (a = aopt, b = bopt)))
plt.xlabel('X',fontsize=12)
plt.ylabel('Y',fontsize=12)
plt.show()
