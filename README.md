# Health-Literacy-COVID19

## Contents
[Introduction](https://github.com/aakula7/Health-Literacy-COVID19/blob/master/README.md#introduction)

## Introduction
Health literacy is a discrete form of literacy and becoming an increasingly important aspect for social, economic, and health development. It is already seen as a crucial tool for the prevention of non-communicable disease with investments in education and communication. However, there is not enough knowledge on health literacy’s impact on communicable diseases. Today with the rapid development of coronavirus disease 2019 (COVID-19), a communicable disease, there has been a need for people to acquire and apply health information. (Paakkari & Okan, 2020) Health communication intended to educate people has become widely available. However, there is also a lot of misinformation, thus forcing individuals to filter and be health literate.

This investigation created successful models for predicting the number of COVID-19 cases from data regarding the United States Census Bureau, Internal Revenue Service (IRS), Centers of Medicare and Medicaid Services, and National Science Board. The successful execution of these machine learning models builds an association between health literacy and COVID-19, under the assumption that states with high COVID-19 cases associate with areas of lower health literacy. These models can be deployed for further analysis of state health care costs and policy challenges.

## Materials
The goal of the research was to develop machine learning models for predicting  COVID-19 case count per state according to the socioeconomic and sociodemographic analysis of each state. This goal was achieved through primary exploratory data analysis (EDA) and several machine learning models consisting of linear regression and support vector regression (SVR), ensemble learning models which are random forest, and xgboost.

### Machine Learning
Machine learning is defined as computational methods, which use available past information to improve performance or make accurate predictions. Learning corresponds to adjusting the values of these parameters so that the model matches best with the data it sees during training. Based on this training data, the model with the help of its hyperparameters becomes specialized to the particular task that underlies the data. This version of the model becomes the algorithm for that task. (Alpaydin, 2010)

### Linear Regression and Support Vector Regression (SVR)
Linear Regression attempts to model the relationship between two variables by fitting a linear equation to the observed data. One variable is considered as the explanatory variable, and the other is considered to be the dependent variable. This does not imply there has to be a cause and effect relationship between the variables, but there is a significant association between the two. (Barron & Kim, 1998)

Support vector machines (SVM) are a set of supervised learning methods used for classification, regression, and outlier detection. For the goal of this project, Support Vector Regression is used. The advantages of SVRs are: effective in high dimensional spaces and where the number of dimensions is greater than the number of samples. SVR models use a subset of training points in the decision function (called support vectors), and different kernel functions can be specified for the decision function. SVRs also depend only on a subset of the training data, because the cost function for building the model does not care about training points that lie beyond the margin. The cost function also ignores samples whose prediction is close to their target. (Steinwart & Christmann, 2008)

### Ensemble Learning
Ensemble learning is the process by which multiple models are trained to solve the same problem. Compared to ordinary machine learning approaches which try to learn one hypothesis from training data. (Zhou, 2009) Ensemble methods construct a set of hypotheses and combine them for more accurate analysis. They are primarily used to improve the performance of a model or reduce the likelihood of an unfortunate selection of a poor one.

An ensemble model is constructed in two steps, where the number of base learners are produced and they are combined with majority voting for classification and weighted averaging for regression. (Zhou, 2009) To produce a good ensemble, the base learners should be as accurate as possible and as diverse as possible. This can be completed through the use of various accuracy estimation processes, such as cross-validation.

The main ensemble methods used are boosting, bagging, and stacking algorithms. In boosting, successive trees give extra weight to points incorrectly predicted by earlier predictors. In the end, a weighted vote is taken for prediction. (Zhou, 2009) Bagging trains a number of base learners from a different bootstrap sample, which is obtained by subsampling the training data set. It then combines the subsampling with a majority voting and the most-voted class is predicted.

### Random Forest
Random forest adds an additional layer of randomness to bagging. (Liaw & Wiener, 2002) In addition to constructing each tree using a different bootstrap sample of the data, random forests change how the trees are constructed. In a random forest tree, each node is split using the best among a subset of predictors randomly chosen at that node. Although it is somewhat counterintuitive, the method turns out to perform very well compared to other algorithms and is robust against overfitting.

### XGBoost
Today XGBoost is one of the most widely used learning algorithms in machine learning due to its adaptability, easy to interpret, and high accuracy features. The tree ensemble model consists of a set of classification and regression trees (CART). This is slightly different from decision trees, in which the leaf only contains decision values. In CART, a real score is associated with each of the leaves. A single tree is not strong enough to be used, therefore an ensemble model is used which includes functions as parameters and cannot be optimized using traditional methods in Euclidean space. Instead the model is trained in an additive manner of the objective function, where the model learns from the functions (fi), each containing the structure of the tree and the leaf scores. (Tianqi & Carlos, 2016)

### Standard Scaler
Feature scaling has a significant impact on machine learning algorithms, especially if the data is skewed giving a bias. Most machine learning algorithms work on the magnitude of the measurement not on the unit. The algorithms use Euclidean distance between data points, therefore the distance of a higher magnitude feature and low magnitude feature would produce undesired results. Data scaling was done by a Z-score algorithm, where the features are scaled according to the mean of zero and standard deviation of 1.  Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. (Misra & Yadav, 2019)

## Data
### COVID-19 Case Count
Data was downloaded from New York times COVID-19 case count tracker per state per day. The data table was parsed to collect the total positive coronavirus case count per state as on 16 July, 2020. All of the factors measuring health literacy were merged with this data set to analyze and build machine learning models, understanding how these factors influence the coronavirus case count.

### Census Demographic
Data was collected from the U.S. Census Bureau on the socio demographics of each state. Census demographic was measured through age, gender, and ethnicity populations per state. Age was broken into the following groups: less than 5 years, 5 to 9 years, 10 to 14 years, 15 to 19 years, 20 to 24 years, 25 to 34 years, 35 to 44 years, 45 to 54 years, 55 to 59 years, 60 to 64 years, 65 to 74 years, 75 to 84 years, and 85 years and older. Gender was broken into male and female. Lastly ethnicities were grouped into white, black or african american, american indian or alaska native, asian, native hawaiian and other pacific islander, or hispanic or latino.

### Education
The second aspect analyzed as a part of sociodemographics, was education of each state. This data was collected from the National Science Foundation.  Education was measured through analyzing public school expenditure per state GDP, public school expenditure per student enrollment, literacy level, and public school teacher salary. Public school expenditure per state GDP and public school expenditure per student enrolled had the same data in regards to public school expenditure. However, they had different calculations in the ratio columns of per state GDP and per student enrolled. Literacy level encompassed fourth grade math performance score, 4th grade science performance score, 8th grade math performance score, 8th grade science performance score, number of ELL students enrolled, labor force with bachelor’s degree, total labor force, total number of doctorate degrees, earned less than a high school diploma, earned a high school diploma, earned some college or associate degree, and earned a bachelor degree.

### Healthcare Access and Cost
Health literacy also includes the access and affordability of healthcare, which was data collected from Centers for Medicare and Medicaid Services. This factor was measured through health expenditure per state, health expenditure per state capita, and health insurance enrollments, health insurance cost per state, and health state premium costs. Healthcare expenditure per state and per state capita encompassed the same categories, which were personal health care, hospital services, physician and clinical services, other professional services, home healthcare, drugs and non-durables, durables, other healthcare, and total healthcare. All of the categories were expenses per state. However, in the state capita data frame the data was divided to the capita of the state. Health insurance enrollment, health insurance cost per state, and health insurance cost per enrollee took into account the three main health insurance categories of medicaid, medicare, and private health insurance. Lastly health state premium costs included employer sponsored health insurance premium cost (single coverage), employer sponsored health insurance premium cost (family coverage), employee contribution to premium costs (single coverage), employee contribution to premium costs (family coverage), employee contribution to premium costs (combined average), employee deductible costs (single coverage), employee deductible costs (combined average), total potential out-of-pocket costs, and overall health insurance premium costs.

### Income
To analyze the socioeconomic factors of health literacy, income was the first factor chosen to be analyzed. This was collected from Internal Revenue Services and the US Census Bureau. The income factor included adjusted gross income, poverty level, and unemployment. Adjusted gross income was divided into top 1%, 5%, 10%, 25%, 50%, and 75% of population per state. Poverty was broken into all ages in poverty (the total number of people under poverty) and the number of people under the small area income and poverty estimate classification per state. The last factor measuring income was the number of people under the following categories, civilian labor force, number unemployed, number employed, and median household income. 

### Jobs
The last factor in the investigation and another socioeconomic factor of health literacy was analyzing the number of people in each type of job, this data was collected from the National Science Foundation. The categories for job identification were computer mathematical scientists, engineers, life scientists, physical scientists, science engineer workers, social scientists, and technical workers.

## Data Web
![Data](https://github.com/aakula7/Health-Literacy-COVID19/blob/master/Data.jpg)

## Method Overview
![Method](https://github.com/aakula7/Health-Literacy-COVID19/blob/master/Method.jpg)

## Discussion
Health literacy is a composite term used to describe the capacities of persons to meet the demands related to health in modern society. Health literacy is crucial for the prevention of non-communicable diseases. However, with the increased chance in pandemics, the next question is how effective health literacy is for communicable diseases such as COVID-19. This research measured health literacy per state and created accurate machine learning models to predict coronavirus cases per state. Therefore showing health literacy can be used to fight the impact of communicable diseases.

### Healthcare Cost
People with limited health literacy have reported poorer overall health, and less likely to utilize preventive screenings adherence to medical regimes. Low health literate individuals are also more likely to be hospitalized, causing a huge financial burden. On the contrary, people who gain improved skills to retrieve updated health information may show a greater demand for anything new but also more expensive services. (Parker et al., 2003)

Communicable diseases such as the seasonal influenza have $2.0-$5.8 billion healthcare costs annually from the 2001/2002 - 2008/2009 flu seasons. The type B influenza virus strain accounted for 37% of healthcare costs across all seasons, and as much as 66% in a single season. (Yan et al., 2017) This is a significant cost that is on a trajectory for exponential growth in cost.

With health care costs not decreasing, and seeing a significant presence of low health literacy in america, the xgboost model built in this investigation can be used to analyze the impact of communicable diseases given each state’s health literacy demographics. Applications of these models can be seen in the private and public sector.
Private investment in U.S. healthcare has grown significantly over the past decade. In 2018 there were about 800 private equity deals, which had a total value of more than $100 billion. One of the main focuses of private equity firms has been buying and growing the specialties that generate a disproportionate share of surprise bills: emergency room physicians, hospitals, anesthesiologists, and radiologists. (Gustafsoon et al., 2019) These are all the main areas of the hospital being used to treat communicable diseases, and very prominent during the fight with the coronavirus. States with low health literacy are seeing the highest COVID-19 case counts as citizens of the state are not able to understand and apply their health literacy to keep themselves safe from the virus. These are the states which will have the highest healthcare costs due to increased usage of emergency rooms, radiologists, respiratory therapists, and many more healthcare services.

Similar to private equity firms, the government needs to allocate financial resources to combat public health risks. In order to successfully keep the case and death count low the government has to make sure each citizen has access to affordable health care. Therefore they can use the machine learning models to analyze which states will be impacted the most by communicable diseases.

### Healthcare Policy Challenges
As discussed, problems with health literacy are costly as millions of Americans struggle to read and understand the information needed to function in the healthcare system. Another issue brought by health literacy, is a policy issue at the intersection of health and education. 

Many health system beneficiaries cannot calculate their need and affordability of supplemental insurances. As an example, today proposal policies for patients’ bills of rights would provide managed care enrollees with access to an external appeals process for disputed claims. Can patients with low health literacy take advantage of this and other rights created under this legislation? (Parker et al., 2003)

Professional medical societies such as American Medical Association have helped to raise awareness by focusing greater attention on health communication and developed guidelines about patients’ understanding and the readability of patient materials. In the effort to improve quality, one goal for health policy should be to ensure a health literate America, creating more informed patients who have better outcomes, as they are more concordant with the people who provide health services. (Parker et al., 2003) This will allow individuals to seek care earlier because they recognize warning signs, read and comprehend instructions, understand what their doctors advise, and they are not afraid to ask questions when they do not understand. The hopes to bring awareness can be merged with our model to understand which states have the lowest health literacy. Those states are also the ones to have the highest number of communicable diseases. This will allow organizations and agencies to help state governments navigate public health policy enforcement and understanding when combating communicable diseases such as COVID-19. Specifically this could help effective implementation of mandatory masks and social distancing, sanitation, necessary lockdown, and other guidelines for the public to understand and follow.

### Limitations
The concept of health literacy has changed significantly over the last couple of decades. Originally it was defined as literacy through reading, writing, and numeracy skills in the health domain. Today it has evolved into a multidimensional concept that is still evolving. This makes it difficult to accurately measure health literacy consistently and accurately build machine learning models. Although this investigation was able to create accurate models, on average the models still had an error of 8000 to 20000. This could lead to inaccurate investments in the health care system of states, where it is not necessary. 

After building, training, testing, and tuning machine learning models, the next step is feature selection, which can reduce computation time, improve learning accuracy, and facilitate a better understanding for the learning model or data. However, with inconsistencies in defining health literacy, there is no knowledge on which factors or sub-factors can be deleted. This has caused the research to include all sub-factors, causing a possibility of noise in the data prediction.

The other issue with analyzing health literacy’s effect on communicable disease is size of the data. As this research considered data per state, there were only 50 rows, but numerous columns of data. This is not ideal as splitting the dataset for training and testing the machine learning model is done on the rows. There the model built used 39 states for train data and only 12 states for testing, which is not enough to show valid and confident results.

## Conclusion
Health literacy skills are those people use to realize their potential in health situations. They apply these skills either to make sense of health information and services. Health literacy and clear communication between health professionals and patients are key to improving health and the quality of healthcare. Health literacy has been instrumental in raising awareness regarding non-communicable diseases. However, experts predict communicable diseases and pandemics such as COVID-19 are on the rise. This has raised the question regarding health literacy’s effect on communicable and higher transmittable diseases. This importance is becoming exponentially important as technology grows and misinformation is spread. We investigated this relationship with the use of machine learning models in hopes of successfully predicting the coronavirus case count per state and accurately classifying each state to having above or below the national coronavirus case count average.

Health literacy was defined and measured through census demographics, education levels, health care costs and access, income levels, and types of jobs. Taking into consideration all of the socioeconomic and sociodemographic factors, we were able to successfully predict coronavirus case count per state with a mean average error of 11787.828 for the best machine learning model, xgboost. Census and education level demographics of health literacy showed an accurate random forest machine learning model, with a mean average errors of 16069.295 and 13529.276 respectively. Healthcare access and cost measurements of health literacy showed an accurate xgboost model, with a mean average error of 12625.241. Finally, income level and type of jobs held aspects of health literacy both showed accurate support vector regression models, with mean average errors of 10609.005 and 21331.564 respectively. Therefore there is a relationship present between health literacy and communicable diseases.

In theory these models are accurate to help the private and public sector in making valuable decisions in regards to investment, healthcare costs, and healthcare policy. However, further research needs to be completed in solidifying the definition and factors encompassing health literacy, which can lead to more accurate models and decisions in health literacy.

## References
### Books
Alpaydın, E. (2010). Introduction to Machine Learning (2nd ed.). Cambridge, MA: MIT Press.

Steinwart, I., & Christmann, A. (2008). Support Vector Machines. New York, NY: Springer Science & Business Media. doi:10.1007/978-0-387-77242-4

Zhou, Z. H. (2009). Ensemble Learning. Encyclopedia of biometrics, 1, 270-273.

### Articles
Coronavirus. (2020). Retrieved from https://www.who.int/health-topics/coronavirus

Barron, A., & Kim, J. (1997). Linear Regression. Retrieved from http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm

Gustafsoon, L., SeerVai S., & Blumenthal D. (2019). The Role of Private Equity in Driving Up Health Care Prices. Retrieved from https://hbr.org/2019/10/the-role-of-private-equity-in-driving-up-health-care-prices#:~:text=Private%20equity%20and%20venture%20capital,of%20more%20than%20%24100%20billion.

### Periodical
Baker, D. W. (2006). The meaning and the measure of health literacy. Journal of general internal medicine. 21(8), 878–883.  Retrieved from https://doi.org/10.1111/j.1525-1497.2006.00540.x

Institute of Medicine (US) Committee on Health Literacy, Nielsen-Bohlman, L., Panzer, A. M., & Kindig, D. A. (Eds.). (2004). Health Literacy: A Prescription to End Confusion. National Academies Press (US). Retrieved from https://doi.org/10.17226/10883.

Liaw, A., & Wiener, M. (2002). Classification and Regression by randomForest. Retrieved from https://www.researchgate.net/profile/Andy_Liaw/publication/228451484_Classification_and_Regression_by_RandomForest/links/53fb24cc0cf20a45497047ab/Classification-and-Regression-by-RandomForest.pdf

Misra, P., & Yadav, S. A. (2019). Impact of Preprocessing Methods on Healthcare Predictions. International Conference on Advanced Computing and Software Engineering (ICACSE). Retrieved from http://dx.doi.org/10.2139/ssrn.3349586

Paakkari L., & Okan O. (2020). COVID-19: Health Literacy is an Underestimated Problem. The Lancet. Public health, 5(5), e249–e250. Retrieved from https://doi.org/10.1016/S2468-2667(20)30086-4

Parker, R. M., Ratzan, S. C., & Lurie, N. (2003). Health literacy: A Policy Challenge for Advancing High-Quality Health Care. Health Affairs (Project Hope), 22(4), 147–153. Retrieved from https://doi.org/10.1377/hlthaff.22.4.147

Tianqi C. & Carlos G. (2016). XGBoost: A Scalable Tree Boosting System. Association for Computing Machinery.  Retrieved from https://doi.org/10.1145/2939672.2939785
Yan, S., Weycker, D., & Sokolowski, S. (2017). US healthcare costs attributable to type A and type B influenza. Human vaccines & immunotherapeutics, 13(9), 2041–2047. Retrieved from https://doi.org/10.1080/21645515.2017.1345400
