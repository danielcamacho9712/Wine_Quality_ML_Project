# Wine Quality Project

## 1. Introduction

This is a set with the compilation of several chemicals parameters related with the wine quality. We are going to use this dataset to predict if the wine is a good quality wine or not, using parameters like:<br/>
1- Fixed acidity<br/>
2- Volatile acidity<br/>
3- Citric acid concentration<br/>
4- Residual sugar<br/>
5- Amount of chlorides<br/>
6- Total amount of sulfur dioxide<br/>
7- Solution density<br/>
8- Solution pH<br/>
9- Amont of sulphates<br/>
10- Alcohol grade<br/>

## 2. Data cleanning and preparation 

The first thing that we checked in the dataset was if there was any missing or nan values present inside the data:<br/>

<div align="center">
  <img src="Images/is_nan.png" alt="Screenshot" width="200">
</div>
<p><strong>Figure 1.</strong> Every Nan values in the dataset

There wasn't nan values present in the dataset. After this we explore the data, to see if all the values were in a logical range:

<div align="center">
  <img src="Images/data_exploration.png" alt="Screenshot1">
</div>
<p><strong>Figure 2.</strong> Dataset column's histograms

We are addressing a classification problem, the desired output labels are bad quality wine that will be represented with a 0 value and good quality wine 
that will be represented with a 1 value. Input features had different scales:<br/>
1- Fixed acidity: <p><strong> 0<x<16 </strong></p>
2- Volatile acidity: <p><strong> 0<x<1.6 </strong></p>
3- Citric acid concentration: <p><strong> 0<x<1 </strong></p> 
4- Residual sugar: <p><strong> 0<x<16 </strong></p>
5- Amount of chlorides: <p><strong> 0<x<0.6 </strong></p>
6- Total amount of sulfur dioxide: <p><strong> 0<x<75 </strong></p>
7- Solution density: <p><strong> 0<x<300 </strong></p>
8- Solution pH: <p><strong> 2.6<x<4.2 </strong></p>
9- Amont of sulphates: <p><strong> 0.25<x<2 </strong></p>
10- Alcohol grade: <p><strong> 8<x<15.2 </strong></p>