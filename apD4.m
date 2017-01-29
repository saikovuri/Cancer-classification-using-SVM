clear all;clc;close all;

load('Final_Transformed.mat')

nTrainSamples = 38;
nTestSamples = 35;

startIndex = 1;
nF = [4 8 16 24 48 72 96 120 144 168 192 216 240 264 288 300];  

kernel_fun = {'linear','rbf'}; 

for i= 1:10 %Number of input data 
    for x = 1:2 %Number of kernel functions 
        for y = 1:size(nF,2) %Number of features 
            nFeatures = nF(y); 
             
            %Reading the training data 
            trainData = TrainLk{1,i}(:,startIndex:startIndex+nFeatures-1); 
            trainClasses=TrainLk{2,i};
            
            %Reading the test data
            testData = TestLk{1,i}(:,startIndex:startIndex+nFeatures-1); 
            testClasses = TestLk{2,i}; 
            
            %Training the SVM 
            svmStruct = svmtrain(trainData,trainClasses,'Kernel_Function',char(kernel_fun(x)),...
                'Autoscale','false');
            
            %Classifying the training data 
            classifiedClassesT = svmclassify(svmStruct, trainData); 
            
            %Calculating the distance of the test sample from the seperating plane
            sv = svmStruct.SupportVectors;
            alphaHat = svmStruct.Alpha; 
            bias = svmStruct.Bias; 
            kfun = svmStruct.KernelFunction; 
            kfunargs = svmStruct.KernelFunctionArgs; 
            numSV = numel(svmStruct.Alpha); 
            svIndices = svmStruct.SupportVectorIndices; 
            
            %Calculating the denominator usinng alpha'*k(sv,sv)*alpha = w*w 
            kx=feval(kfun,sv,sv,kfunargs{:}); 
            denom=sqrt(alphaHat'*kx*alphaHat); 
            
            %Calculating the margin as mean of the distances of the support vectors to the seperating hyperplane 
            margin = 0; 
            for s = 1:numSV 
                index = svIndices(s,1); 
                dist = abs((feval(kfun,sv,sv(s,:),kfunargs{:})'*alphaHat(:)) + bias)/denom; 
                if(trainClasses(index,1)==classifiedClassesT(index,1)) 
                    margin = margin + dist; 
                else
                    margin = margin - dist; 
                end
            end
            margin = margin/numSV; 
            
            %Classifying the testdata using svmclassify 
            classifiedClasses = svmclassify(svmStruct,testData); 
            
            %Calculating the performance of classifier 
            cp = classperf(testClasses, classifiedClasses); 
            calP(x,y)=cp.CorrectRate;
            Acc{1,i}=calP;

            for k=1:nTestSamples 
                dist = (feval(kfun,sv,testData(k,:),kfunargs{:})'*alphaHat(:)) + bias; 
                sampleClass = sign(dist); 
                dist=abs(dist)/denom; 
                
                %Writing the distance to output file 
                if (k==1)
                    D_outi=[dist margin classifiedClasses(k,1) sampleClass testClasses(k)];
                else
                D_outi=vertcat(D_outi ,[dist margin classifiedClasses(k,1) sampleClass testClasses(k)]);
                end
            end
            D_out{x,y}=D_outi;
            D_output{1,i}=D_out;
        end
    end
    i
end

%

for i= 1:10 %Number of input data 
    if (i==1)
        Tot_AccL =Acc{1,1}(1,:);
    else
        Tot_AccL=vertcat(Tot_AccL ,Acc{1,i}(1,:));
    end
end
for i= 1:10 %Number of input data 
    if (i==1)
        Tot_AccR =Acc{1,1}(2,:);
    else
        Tot_AccR=vertcat(Tot_AccR ,Acc{1,i}(2,:));
    end
end


%% Plot Classification Accuarcy

figure;boxplot(Tot_AccL,'labels',nF);
title('Classification Accuarcy of Linear SVM kernel - Leukemia Dataset');xlabel('Number of top genes');ylabel('Accuarcy Perscentage');
% Convert y-axis values to percentage values by multiplication
     a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
% Create a vector of '%' signs
     pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
     new_yticks = [char(a),pct];
% 'Reflect the changes on the plot
     set(gca,'yticklabel',new_yticks) 
figure;boxplot(Tot_AccR,'labels',nF);
title('Classification Accuarcy of RBF SVM kernel - Leukemia Dataset');xlabel('Number of top genes');ylabel('Accuarcy Perscentage');
% Convert y-axis values to percentage values by multiplication
     a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
% Create a vector of '%' signs
     pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
     new_yticks = [char(a),pct];
% 'Reflect the changes on the plot
     set(gca,'yticklabel',new_yticks) 
     
%  M = mean(Tot_Acc);

%% Plot Margin values

for i=1:10
    for j=1:size(nF,2)
    MargL(i,j)=D_output{1,i}{1,j}(1,2);
    end
end
figure;boxplot(MargL,'labels',nF);
title('Margin values of linear SVM kernel - Leukemia Dataset');xlabel('Number of top genes');ylabel('Margin Values');

for i=1:10
    for j=1:size(nF,2)
    MargR(i,j)=D_output{1,i}{2,j}(1,2);
    end
end
figure;boxplot(MargR,'labels',nF);
title('Margin values of RBF SVM kernel - Leukemia Dataset');xlabel('Number of top genes');ylabel('Margin Values');

%% Plot Rate of change of Margin wrp No. genes 

ML= mean(MargL);
for j=1:size(nF,2)
  if (j==1)
      RateML(j)=ML(j)/nF(j);
  else
      RateML(j)=(ML(j)-ML(j-1))/(nF(j)-nF(j-1));
  end
end
figure;plot(RateML,'-s');
title('Rate of change of Margin wrp No. genes (linear SVM kernel)');xlabel('Number of top genes');ylabel('d(Margin Values)/d(genes)');


MR= mean(MargR);
for j=1:size(nF,2)
  if (j==1)
      RateMR(j)=MR(j)/nF(j);
  else
      RateMR(j)=(MR(j)-MR(j-1))/(nF(j)-nF(j-1));
  end
end
figure;plot(RateMR,'-s');
title('Rate of change of Margin wrp No. genes (RBF SVM kernel)');xlabel('Number of top genes');ylabel('d(Margin Values)/d(genes)');

% %% Rate of change of Accuarcy wrp No. genes
% 
% MAccL= mean(Tot_AccL);
% for j=1:size(nF,2)
%   if (j==1)
%       RateAccL(j)=MAccL(j)/nF(j);
%   else
%       RateAccL(j)=(MAccL(j)-MAccL(j-1))/(nF(j)-nF(j-1));
%   end
% end
% plot(RateAccL,'-s');
% title('Rate of change of Accuarcy wrp No. genes (linear SVM kernel)');xlabel('Number of top genes');ylabel('d(Accuarcy)/d(genes)');

%% Missclassified linear 
for i=1:10
    for j=1:size(nF,2)
        missL{i,j}=find((D_output{1,i}{1,j}(:,3))-(D_output{1,i}{1,j}(:,5))~=0); % Index of missclassified linear
        missLV{i,j}=D_output{1,i}{1,j}(missL{i,j}(:,1)); % % Values of missclasified linear
        mrgL(i,j)=D_output{1,i}{1,j}(1,2); %Margin values of linear
        miss{i,j}= [missL{i,j} missLV{i,j}] % Index and values of missclassified linear 
    end
end

%% Missclassified RBF
for i=1:10
    for j=1:size(nF,2)
        missR{i,j}=find((D_output{1,i}{2,j}(:,3))-(D_output{1,i}{2,j}(:,5))~=0); % Index of missclassified RBF
        missRV{i,j}=D_output{1,i}{2,j}(missR{i,j}(:,1)); % % Values of missclasified RBF
        mrgR(i,j)=D_output{1,i}{2,j}(1,2); %Margin values of RBF
        missR{i,j}= [missR{i,j} missRV{i,j}] % Index and values of missclassified RBF 
    end
end
