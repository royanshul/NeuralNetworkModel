setwd("C:/Users/ansroy/Desktop/Personal/IIIT B/Assignment & Case Study/Assignment - Neural Network")
pkgs <- c("MASS","car","caret","h2o","ggplot2") 
for (pkg in pkgs) { 
   if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkgs) } } 
#  #The following two commands remove any previously installed H2O packages for R. Use them un case of any issues in h2o initialisation.
#  #if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) } 
#  #if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") } 
#  #The following code, execute from command prompt to start h2o package if not starting from R. & then execute h2o.init & it will merge with the command prompt cluster
#  #"C:\Program Files\Java\jdk1.8.0_111\bin\java.exe" -Xmx30g -jar "C:\Users\ansroy\Documents\R\win-library\3.3\h2o\java\h2o.jar" -nthreads 2
#  #Find the processid being used from command prompt in admin mode:
#  #netstat -aon |find /i "listening" |find "54322"
#  # Kill the ProcessID 
#  #taskkill /F /PID 3312
library("car")
library("caret")
library("MASS")
library("h2o")
library(caTools)
#Plotting all the graphs in PDF
#Data Undertanding and Preparation of Master File
#telecom_churn <- h2o.importFile("telecom_nn_train.csv")
telecom_churn <- read.csv("telecom_nn_train.csv")
str(telecom_churn)
summary(telecom_churn)



#Data Preparation
sum(is.na(telecom_churn))   # 6 NA Values
sapply(telecom_churn,function(x){sum(is.na(x))})  #Shows that for Total Charges columns, there are NA Values.
telecom_churn$TotalCharges[is.na(telecom_churn$TotalCharges)]<- 0
sum(is.na(telecom_churn))  #THe NA values has been succesfully removed.
nrow(telecom_churn)
str(telecom_churn)
summary(telecom_churn)
nrow(telecom_churn[duplicated(telecom_churn),])  #10 duplicates 
telecom_churn <- telecom_churn[!duplicated(telecom_churn),]



#EDA
#Function for creating plot. 
telecom_churn_bargraph <- function(z, na.rm = TRUE, ...) {   nm <- names(z)
for (i in seq_along(nm)) {
  plots <-ggplot(z,aes_string(x=nm[i],fill=factor(z$Churn))) + geom_bar(position = "fill")+
    guides(fill=guide_legend(reverse=TRUE))+
    scale_fill_discrete(labels=c("Good Customer","Churned Customer"))+
    labs(fill='churn status')
  ggsave(plots,width = 20, height = 8, units = "cm",filename=paste("myplot",nm[i],".png",sep=""))
}
}
# 
# #Plots created will be saved in the working directory. Plots are created
# #for all categorical variables. 
telecom_churn_bargraph(telecom_churn[,-c(5,8,9)])

# Analysis on numeric data
ggplot(telecom_churn, aes(x= telecom_churn$Churn, y= telecom_churn$tenure)) + geom_boxplot()
# A person who is churing, their tenure are comparatively lesser as compare to those who are not churning

ggplot(telecom_churn, aes(x= telecom_churn$Churn, y= telecom_churn$MonthlyCharges)) + geom_boxplot()
# Monthly charges for churning customers are higher as compare to non churning customers

ggplot(telecom_churn, aes(x= telecom_churn$Churn, y= telecom_churn$TotalCharges)) + geom_boxplot()
# Total charges for churning customers are lesser than compared to non churning customers





#Outliers Treatment 
#Tenure
boxplot(telecom_churn$tenure)
quantile(telecom_churn$tenure,seq(0,1,0.01))   #Outlier treatment not required

#TotalCharges 
boxplot(telecom_churn$TotalCharges)
quantile(telecom_churn$TotalCharges,seq(0,1,0.01))  #Outlier treatment not required
#MonthlyCharges
boxplot(telecom_churn$MonthlyCharges)
quantile(telecom_churn$MonthlyCharges,seq(0,1,0.01)) #Outlier treatment not required
telecom_churn$Churn <- as.character(telecom_churn$Churn)



#telecom_churn$Churn[telecom_churn$Churn== 'Yes'] <- '1' 
#telecom_churn$Churn[telecom_churn$Churn== 'No'] <- '0' 
#telecom_churn$Churn <- as.integer(telecom_churn$Churn)
telecom_churn$Churn <- as.factor(telecom_churn$Churn)


#Neural networks give good results
#when continous variables are scaled.
#so scaling of continous variables are undertaken
telecom_churn[ ,c(5,8,9)] <- scale(telecom_churn[ ,c(5,8,9)])

#validation and train dividing
set.seed(1000)
split_indices <- sample.split(telecom_churn$Churn,SplitRatio = 0.7)
#split_indices
train <-telecom_churn[split_indices==T,]
validation <-telecom_churn[split_indices==F,]

#The split datasets are written to the working directory
#for importing as H2o objects
write.csv(train, file='churnTrain.csv', row.names=FALSE)
write.csv(validation,file='churnvalidation.csv',row.names=FALSE)


h2o.init(nthreads=-1,min_mem_size = "3g")
h2o.getConnection()


#Import the train and validation datasets as H2o objects
train.h2o <- h2o.importFile("churnTrain.csv")
validation.h2o <-h2o.importFile("churnvalidation.csv")
#class(train.h2o)

colnames(train.h2o)  #InDependant Variables - 30 & one is the final result Churn (Dependant Vriable)
# Specify the response and predictor columns
y.dep <- "Churn"
x.indep <-setdiff(names(train), y.dep)

####################################################################
################### (Neural network without epoch) ####################

#Create an initial model. one hidden layer with number of neurons equal to 3 times
#number of inputs give good results, the same is tried and results checked


model_1 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(90),
                            hidden_dropout_ratio = c(0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)

# View the specified parameters of your deep learning model
model_1@parameters
model_1  # display all performance metrics
#Check the confusionMatrix values for 
h2o.performance(model_1, train = TRUE) # training set metrics
h2o.performance(model_1,valid  = TRUE)
#sensitivity <- 376/1450


#Performance Metrics H2o >
#Training Data -> MSE:  0.5050864,R^2:  -1.020346,LogLoss:  16.43209,AUC:  0.4738132,Gini:  -0.05237369,Accuracy :0.500098 , Specificity:0.118339
#Validation Data -> MSE:  0.2644901,R^2:  -0.3764585,LogLoss:  4.812506,AUC:  0.761191,Gini:  : 0.5223819 Accuracy : 0.779810, sensitivity - 0.479,Specificity:0.893870
#specificity : 19/26 = 0.730
#Because the Accuracy is not good and all other values are not good, BIAS is high. we are making changes to the model by increasing complexity.



#Single layer with 500 Neurons
model_2 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(500),
                            hidden_dropout_ratio = c(0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)
#Check the confusionMatrix values for 
h2o.performance(model_2, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_2,valid  = TRUE)
#sensitivity <- 259/1183

#Training ConfusionMatirx -> MSE:  0.2309991,R^2:  0.07600351,LogLoss:  5.771186,AUC:  0.7996605,Gini:  0.599321,Accuracy :0.766503,Specificty ;0.827586
#Validation Confusion Matrix -> MSE:  0.2751111,R^2:  -0.4317325,LogLoss:  6.973087,AUC:  0.7736581,Gini:  0.5473163, Accuracy :0.753388, sensitivity : 0.44287,specificity:0.796889
#Because the Accuracy is not good and all other values are not good, BIAS is high. we are making changes to the model by increasing complexity.


#Single layer with 500 Neurons
model_3 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(500,500,500),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_3, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_3,valid  = TRUE)
#sensitivity <- 383/1476

#Training ConfusionMatirx -> MSE:  0.2180215,R^2:  0.1279138,LogLoss:  7.530197,AUC:  0.7826211,Gini:  0.5652421, Accuracy:0.781978, Specificty:0.728056
#Validation Confusion Matrix -> MSE:  0.2879403,R^2:  -0.4984984,LogLoss:  9.928826,AUC:  0.735454,Gini:  0.4709079,Accuracy : 0.781978, sensitivity : 0.4680187,Specificity:0.688015
#There is no much difference. BIAS is very high.

#Lets change Activation Function now.
model_4 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="RectifierWithDropout",
                            hidden = c(500,500,500),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_4, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_4,valid  = TRUE)
#sensitivity <- 383/1476

#Training Set-> MSE:  0.2924553,R^2:  -0.1698212,LogLoss:  10.07248,AUC:  0.7209239,Gini:  0.4418479,Accuracy :0.718315, Specificity:0.880094
#Validation Set ->MSE:  0.2506775,R^2:  -0.3045753,LogLoss:  8.658094,AUC:  0.6601146,Gini:  0.3202291, Accuracy:0.749322 , Sensitivity:0.518,Specificity:0.844465
#slight increase in sensitivity .. The model is BIAS with high values. Will create new model will complex hyperparameters.


#Keeping the activiztion function and changing the hidden values
model_5 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="RectifierWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_5, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_5,valid  = TRUE)
#sensitivity: 379/1437

#Training Data :MSE:  0.2527691,R^2:  -0.01107624,LogLoss:  8.604558,AUC:  0.7882355,Gini:  0.576471, Accuracy:0.770029, Specificity:0.798197
#validationing Data :MSE:  0.250562,R^2:  -0.303974,LogLoss:  8.49226,AUC:  0.7317967,Gini:  0.4635934, Acuracy:0.752033, sensitivity:0.51509,Specificity:0.784995
#Still there is no good impact on the values.



#Changing the activiztion function 
model_6 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=1000,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.1,0.1,0.1),
                            l1 = 1e-5,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,
                            epochs = 1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_6, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_6,valid  = TRUE)
#sensitivity <- 370/1342 = 0.275

#Training Data :MSE:  0.2176296,R^2:  0.1294817,LogLoss:  7.501234,AUC:  0.7891302,Gini:  0.5782603, Accuracy:0.783154, Specificity:0.724922
#validationing Data :MSE:  0.2856706,R^2:  -0.4866861,LogLoss:  9.847715,AUC:  0.7423254,Gini:  0.4846507, Acuracy:0.714770, Sensitivity: 0.4708589,Specificty:0.684355
#Slightly Downgraded then previous verison. But still BIAS is high.


#Changing the hidden dropout
model_7 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=123,
                            variable_importances = TRUE,
                            activation="MaxoutWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.15,0.15,0.15),
                            l1 = 1e-7,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,epoch=1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_7, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_7,valid  = TRUE)
#sensitivity <- 383/1476

#Training Data :MSE:  0.2214712,R^2:  0.1141151,LogLoss:  7.626624,AUC:  0.7816068,Gini:  0.5632137, Accuracy:0.779518, Specificity:0.720611
#validationing Data :MSE:  0.2737127,R^2:  -0.4244552,LogLoss:  9.453703,AUC:  0.7444048,Gini:  0.4888096, Acuracy:0.726287,Sensitivity: 0.4830918,Specificity:0.706313
#Slightly impreoved then previous verison, Sensitivity dropped little. But still BIAS is high.


#Changing the activation function to TanhwithDropout
model_8 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=123,
                            variable_importances = TRUE,
                            activation="TanhWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.15,0.15,0.15),
                            l1 = 1e-7,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,epoch=1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_8, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_8,valid  = TRUE)
#sensitivity <- 269/739

#Training Data :MSE:  0.2791514,R^2:  -0.1166061,LogLoss:  2.264137,AUC:  0.766135,Gini:  0.53227, Accuracy:0.705306, Specificity:0.929859
#validationing Data :MSE:  0.2593996,R^2:  -0.3499667,LogLoss:  2.174502,AUC:  0.7519284,Gini:  0.5038567, Acuracy:0.768293, Sensitivity: 0.4381107,Specificity:0.938701
# Improve in the performance in terms of Specificity but reduced in Sensitivity sby changing the activation function to TanhwithDropout


#Changing the hidden dropout ratio to check the changes
model_9 <- h2o.deeplearning(x=x.indep,
                            y=y.dep,
                            training_frame = train.h2o,validation_frame = validation.h2o,
                            seed=123,
                            variable_importances = TRUE,
                            activation="TanhWithDropout",
                            hidden = c(200,200,200),
                            hidden_dropout_ratio = c(0.2,0.2,0.2),
                            l1 = 1e-7,
                            nfolds=5,
                            initial_weight_distribution = "Normal",
                            balance_classes = TRUE,
                            sparse=TRUE,
                            reproducible = TRUE,epoch=1,
                            keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_9, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_9,valid  = TRUE)
#sensitivity <- 273/759

#Training Data :MSE:  0.2583324,R^2:  -0.03333011,LogLoss:  1.852602,AUC:  0.7718197,Gini:  0.5436395, Accuracy:0.718817, Specificity:0.935737
#validationing Data :MSE:  0.2487262,R^2:  -0.29442,LogLoss:  1.802118,AUC:  0.7591832,Gini:  0.5183663, Acuracy:0.761518, Sensitivity: 0.4492754,specificity:0.951510
#Slight  Improve in the performance by changing the hidden dropout ratio.

#Model_4 is the best Model created as of now  having max sensitivity, though the sensitivity is not good.
#Looks like this is the best sensitivity we can get with epoch fixed


#################################################################################
########## (Modeling with epoch) ##########################################
#Checkpoint 2: Model - Neural Networks - Tuning hyperparameters WITH epochs:
model_1_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(90),
                                 hidden_dropout_ratio = c(0.1),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_1_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_1_epoch,valid  = TRUE)
#sensitivity <- 288/554

#Training Data :MSE:  0.1632414,R^2:  0.3470346,LogLoss:  0.5534127,AUC:  0.8597076,Gini:  0.7194152, Accuracy:0.787463
#validationing Data :MSE:  0.1899114,R^2:  0.01166346,LogLoss:  0.7056923,AUC:  0.8116378,Gini:  0.6232756, Acuracy:0.731030, Sensitivity: 0.5198556
#Improve in the performance by changing the epoch=20 & hidden =90. Sensitivy has increased with increase in accuracy and reduction in MSE. 
#But with huge BIAS.


model_2_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=123,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(200),
                                 hidden_dropout_ratio = c(0.1),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_2_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_2_epoch,valid  = TRUE)
#sensitivity <- 282/585

#Training Data :MSE:  0.2083473,R^2:  0.1666104,LogLoss:  1.981826,AUC:  0.8466605,Gini:  0.6933209, Accuracy:0.783435,Specificity:0.949451
#validationing Data :MSE:  0.2319844,R^2:  -0.2072927,LogLoss:  2.685686,AUC:  0.781165,Gini:  0.5623299, Acuracy:0.784553, Sensitivity: 0.5,Specificity:0.946935
#Performance degraded by increasing the number of hidden layer


model_3_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.1,0.1,0.1),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_3_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_3_epoch,valid  = TRUE)
#sensitivity <- 268/582

#Training Data :MSE:  0.1833497,R^2:  0.2666013,LogLoss:  6.308642,AUC:  0.823099,Gini:  0.6461981, Accuracy:0.820960
#validationing Data :MSE:  0.2906504,R^2:  -0.2906504,LogLoss:  10.03871,AUC:  0.7062281,Gini:  0.4124562, Acuracy:0.709350, Sensitivity: 0.460
#validation Accuracy- 70.9350%, senstivity - 46%, Misclassification error - 18.3% 
#As the validation error is high the bias of the model is high and the 
#difference between train and validation is 11% the variance is also need
#to be adjusted. First bias is reduced in the next model by increasing complexity 


model_4_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 20,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_4_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_4_epoch,valid  = TRUE)
#sensitivity <- 263/503

#Training Data :MSE:  0.2012491,R^2:  0.1950036,LogLoss:  6.940242,AUC:  0.8045928,Gini:  0.6091856, Accuracy:0.801371
#validationing Data :MSE:  0.2493225,R^2:  -0.2975235,LogLoss:  8.585402,AUC:  0.7288346,Gini:  0.4576691, Acuracy:0.752033, Sensitivity: 0.522286
#The Sensitivity is the best till now with good MSE & Accuracy. We will try to increase it even more by changing hyperparameters.


model_5_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 10,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_5_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_5_epoch,valid  = TRUE)
#sensitivity <- 292/587

#Training Data :MSE:  0.2072478,R^2:  0.1710088,LogLoss:  7.15631,AUC:  0.7961541,Gini:  0.5923083, Accuracy:0.792752
#validationing Data :MSE:  0.2608401,R^2:  -0.3574635,LogLoss:  9.009098,AUC:  0.7474434,Gini:  0.4948868, Acuracy:0.739160, Sensitivity: 0.497
#Sensitivity reduced.

model_6_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="RectifierWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 100,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_6_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_6_epoch,valid  = TRUE)
#sensitivity <- 222/433

#Training Data :MSE:  0.2412385,R^2:  0.03504616,LogLoss:  8.170123,AUC:  0.7816414,Gini:  0.5632829, Accuracy:0.760431,Specificity:0.678292
#validationing Data :MSE:  0.3139169,R^2:  -0.6336856,LogLoss:  10.68437,AUC:  0.7186463,Gini:  0.4372926, Acuracy:0.690379, Sensitivity: 0.4441088,Specificity:0.663312
#The accuracy, sensitivity etc reduced.

#Chaning the activation function to TanhwithDropout
model_7_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="TanhWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 10,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_7_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_7_epoch,valid  = TRUE)
#sensitivity <- 291/597

#Training Data :MSE:  0.2036018,R^2:  0.1855929,LogLoss:  0.9009967,AUC:  0.8241026,Gini:  0.6482052, Accuracy:0.756317
#validationing Data :MSE:  0.2380552,R^2:  -0.2388861,LogLoss:  1.07978,AUC:  0.7947191,Gini:  0.5894381, Acuracy:0.748645, Sensitivity: 0.487
#None of the above model, we get sensitivit of more than 55%. Lets change the epoch value and see.
#Till now, model 4 is the best model. Lets use the same model and increase the epoch and see how it responds.

model_8_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 100,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_8_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_8_epoch,valid  = TRUE)
#sensitivity <- 260/503

#Training Data :MSE:  0.2012491,R^2:  0.1950036,LogLoss:  6.940242,AUC:  0.8045928,Gini:  0.6091856, Accuracy:0.801371
#validationing Data :MSE:  0.2493225,R^2:  -0.2975235,LogLoss:  8.585402,AUC:  0.7288346,Gini:  0.4576691, Acuracy:0.752033, Sensitivity: 0.522286



model_9_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="RectifierWithDropout",
                                 hidden = c(400,400,400),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=5,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 50,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_9_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_9_epoch,valid  = TRUE)
#sensitivity <- 294/662

#Training Data :MSE:  0.2412385,R^2:  0.03504616,LogLoss:  8.170123,AUC:  0.7816414,Gini:  0.5632829, Accuracy:0.760431
#validationing Data :MSE:  0.3139169,R^2:  -0.6336856,LogLoss:  10.68437,AUC:  0.7186463,Gini:  0.4372926, Acuracy:0.690379, Sensitivity: 0.444


model_10_epoch<- h2o.deeplearning(x=x.indep,
                                 y=y.dep,
                                 training_frame = train.h2o,validation_frame = validation.h2o,
                                 seed=1000,
                                 variable_importances = TRUE,
                                 activation="MaxoutWithDropout",
                                 hidden = c(200,200,200),
                                 hidden_dropout_ratio = c(0.2,0.2,0.2),
                                 l1 = 1e-5,
                                 nfolds=10,
                                 initial_weight_distribution = "Normal",
                                 balance_classes = TRUE,
                                 sparse=TRUE,
                                 reproducible = TRUE,
                                 epochs = 50,
                                 keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_10_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_10_epoch,valid  = TRUE)
#sensitivity <- 294/662

#Training Data :MSE:  0.2174441,R^2:  0.1302237,LogLoss:  7.438165,AUC:  0.8121345,Gini:  0.6242689, Accuracy:0.798629
#validationing Data :MSE:  0.2417541,R^2:  -0.2581363,LogLoss:  8.229574,AUC:  0.7440752,Gini:  0.4881503, Acuracy:0.760163, Sensitivity: 0.5264,specificity:0.89
#Sensitivity is still not good. We are trying to increase sensitivity by further reducing the number of neurons.


model_11_epoch<- h2o.deeplearning(x=x.indep,
                                  y=y.dep,
                                  training_frame = train.h2o,validation_frame = validation.h2o,
                                  seed=1000,
                                  variable_importances = TRUE,
                                  activation="MaxoutWithDropout",
                                  hidden = c(100,100,100),
                                  hidden_dropout_ratio = c(0.15,0.15,0.15),
                                  l1 = 1e-5,
                                  nfolds=10,
                                  initial_weight_distribution = "Normal",
                                  balance_classes = TRUE,
                                  sparse=TRUE,
                                  reproducible = TRUE,
                                  epochs = 50,
                                  keep_cross_validation_predictions = TRUE)

#Check the confusionMatrix values for 
h2o.performance(model_11_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns the probability for each class
h2o.performance(model_11_epoch,valid  = TRUE)
#sensitivity <- 294/662

#Training Data :MSE:  0.2116723,R^2:  0.1533109,LogLoss:  6.975641,AUC:  0.8109065,Gini:  0.621813, Accuracy:0.790402
#validationing Data :MSE:  0.3010788,R^2:  -0.5668735,LogLoss:  9.862424,AUC:  0.7497319,Gini:  0.4994637, Acuracy:0.708672, Sensitivity: 0.46


model_12_epoch <- h2o.deeplearning(x=x.indep,
                                   y=y.dep,
                                   training_frame = train.h2o,validation_frame = validation.h2o,
                                   seed=123,
                                   variable_importances = TRUE,
                                   activation="MaxoutWithDropout",
                                   hidden = c(400,400),
                                   hidden_dropout_ratios = c(0.2,0.2),
                                   l1 = 1e-04,
                                   l2 = 1e-05,
                                   max_w2 = 0.1,
                                   epochs = 10,
                                   initial_weight_distribution = "Normal",
                                   balance_classes = TRUE,
                                   sparse=TRUE,
                                   reproducible = TRUE)
#Check the confusionMatrix values for 
h2o.performance(model_12_epoch, train = TRUE) # training set metrics
# Perform classification on the validation set (predict class labels)
# This also returns th probability for each class
h2o.performance(model_12_epoch,valid  = TRUE)
#sensitivity <- 263/503

#Training Data :MSE:  0.2012491,R^2:  0.1950036,LogLoss:  6.940242,AUC:  0.8045928,Gini:  0.6091856, Accuracy:0.801371
#validationing Data :MSE:  0.2493225,R^2:  -0.2975235,LogLoss:  8.585402,AUC:  0.7288346,Gini:  0.4576691, Acuracy:0.752033, Sensitivity: 0.51689
#The sensitivity is still not increasing. 

#Checkpoint 3: Model  - Neural Networks - Best Model
#The best model without epoch is the following:-
#model_4
#Validation - Accuracy:0.749322 , Sensitivity:0.518,Specificity:0.844465
#As the objective is to predict churn, sensitivity needs to be high
#This model has got the highest senstitivity. so this is considered the best

#The best model with epoch is the following:-
#model_10_epoch
#Validation :Accuracy:0.760163, Sensitivity: 0.5264,specificity:0.89
#As the objective is to predict churn, sensitivity needs to be high
#This model has got the highest senstitivity. so this is considered the best

#Overall best model is:-
#model_10_epoch
#Compared to two best models, one with and one without epoch,
#This model's sensitivity is high. So this is considered as the
#overall best

#The variables which are important for the sake of prediction is
#saved as a dataframe in the decreasing order of importance
varimp_model_10_epoch <- as.data.frame(h2o.varimp(model_10_epoch))



h2o.shutdown()

############### END OF Assignment ####################################
