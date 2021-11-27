
%%%%%%%%%%%%%%%%%%%%% 18ETEC004039 %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% HRUTHWIK B C %%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all;
%%%%%%%%%%%%%%%Variable Declaration%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=5; %no of inputs
m=8; %no of neurons in 1st hidden layer
h=4; %no of neurons in 2nd hidden layer
k=1; %no of output layer neuron


w=0.2*rand(m,n)-0.1; %weights of 1st hidden layer
v=0.2*rand(h,m)-0.1; %weights of 2nd hidden layer
u=0.2*rand(k,h)-0.1; %weights of output layer
delw=zeros(m,n);
delv=zeros(h,m);
delu=zeros(k,h);
load mgdata.dat;
inp=mgdata(:,2);
x=1:1:1201;
y_out=zeros(1,1201);
for t1=1:24
 y_out(t1)=inp(t1);
end

wg=0.1; %weight gain
wm=0.95;
b=1;

%%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for epoch=1:10000
 for t = 19:594
 wm=wm*t/594;
 x1=inp(t-18);
 x2=inp(t-12);
 x3=inp(t-6);
 x4=inp(t);
 vec_x=[x1;x2;x3;x4;b];
 a=w*vec_x;
 c=1./(1+exp(-a));
 d=v*c;
 e=1./(1+exp(-d));
 y_out(t+6)=u*e;
 y_des=inp(t+6);


 ey=y_des-y_out(t+6);
 ee=u' * ey;
 ed=e.*(1-e).*ee;
 ec=v' * ed;
 ea=c.*(1-c).*ec;
 error(t)=ey;
 delu=wg*ey.*e' + wm*delu;
 delv=wg*ed*c' + wm*delv;
 delw=wg*ea*vec_x' + wm*delw;
 u=u+delu;
 v=v+delv;
 w=w+delw;
 end
 train_mse(epoch) = mean((error).^2);
end
%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t = 595:1195
 z1=inp(t-18);
 z2=inp(t-12);
 z3=inp(t-6);
 z4=inp(t);
 vec_z=[z1;z2;z3;z4;b];
 a1=w*vec_z;
 c1=1./(1+exp(-a1));
 d1=v*c1;
 e1=1./(1+exp(-d1));
 y_out(t+6)=u*e1;
 y_test(t+6)=u*e1;
 err(t)=inp(t+6)-y_test(t+6);
end
test_mse=mean((err).^2);


%%%%%%%%%%%%%%%Output Plotting%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
time = mgdata(:,1);
z = mgdata(:, 2);
figure(1)
plot(time,z)
title('Mackey-Glass Chaotic Time Series')
xlabel('Time (sec)')
ylabel('x(t)')
figure;
plot(x,inp,'r-',x,y_out,'b:',x,y_test,'g:','lineWidth',2)
axis([0 1201 0.2 1.4]);
grid on
xlabel('Input to the Neuron')
ylabel('Output of the Neural Network')
legend('Desired Output','Training Output','Testing Output','Location','North')
title('Mackey-Glass Time Series Prediction')
figure;
plot(train_mse,'b');
grid on
xlabel('epochs')
ylabel('mse train error')
title('Error during training')
fprintf(1,'mse test error is %f',test_mse)
