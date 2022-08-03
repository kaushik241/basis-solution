import streamlit as st
import numpy as np 
from itertools import combinations
import pandas as pd
from st_aggrid import AgGrid

st.title('Linear Equation Solver')
st.write('This is the website to solve different basis soluiton for  the linear system of equtions. You have to upload the excel file with the coefficient and you can get your answer.')
st.write('Note here that index will start with 0 means x0,x1,x2 like that and right side on the equal to be the last column of excel file')

st.sidebar.title('Linear Equation Solver')

uploaded_file = st.sidebar.file_uploader('Choose a file')


def square_matrix(A,Ab,b):
    solution = np.nan
    if np.linalg.matrix_rank(A) == np.linalg.matrix_rank(Ab):
        if np.linalg.matrix_rank(A) == np.shape(A)[1]:
            solution = np.linalg.solve(A,b)
            solution = [ '%.2f' % elem for elem in solution ]
        elif np.linalg.matrix_rank(A) < np.shape(A)[1]:
            solution = 'Infinitely Many Solution Exists'

        else:
            solution = 'Solution does not exist'

    else:
        solution = 'Solution does not exist'
        
    return solution


if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=None)
    A = df[df.columns[:-1]]
    b = df[df.columns[-1]]
    A = np.array(A) 
    b = np.array(b)

    if np.shape(A)[0] == np.shape(A)[1]:
        
        c = np.reshape(b,(np.shape(b)[0],1))
        np.append(A,c, axis = 1)
        b_t = np.reshape(b,(np.shape(b)[0],1))
        Ab = np.append(A,b_t, axis = 1)

        solution = square_matrix(A,Ab,b)
            
        solution_dict  = {'comb':[], 'solution':[]}
        solution_dict['comb'].append('All')
        solution_dict['solution'].append(solution)
            
    else:
        if np.shape(A)[0] < np.shape(A)[1]:
            solution_dict  = {'comb':[], 'solution':[]}
            no_of_variables_list = list(range(np.shape(A)[1]))
            column_comb = list(combinations(no_of_variables_list, np.shape(A)[0]))
            
            for i in range(len(column_comb)):
                temp = A.transpose()[list(column_comb[i])].transpose()
                
                c = np.reshape(b,(np.shape(b)[0],1))
                np.append(temp,c, axis = 1)
                b_t = np.reshape(b,(np.shape(b)[0],1))
                Ab = np.append(temp,b_t, axis = 1)

                solution = square_matrix(temp,Ab,b)
                
                u = np.shape(A)[1] - np.shape(A)[0]

                zero_value_list = list(set(list(range(np.shape(temp)[1]+ u))) - set(column_comb[i]))
                for j in zero_value_list:
                    solution = np.insert(solution,j,0)
                
                
                
                solution_dict['comb'].append(column_comb[i])
                solution_dict['solution'].append(solution)
                
            
    ans_df = pd.DataFrame(solution_dict)
    AgGrid(ans_df, height=500, fit_columns_on_grid_load=True)
    #st.dataframe(data=ans_df)
    st.write('Made by Kaushik')



            