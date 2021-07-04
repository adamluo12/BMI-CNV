% load dataset n:number of loci g:number of repeated samples
dt=load('D:\Desktop\bayesian_applycation\data_noexonsnp\lrr_22.txt');
cluster=load('D:\Desktop\bayesian_applycation\data_noexonsnp\cluster_22.txt');
n=size(dt,2);
g=size(dt,1);

%data standazation
%dtmean=mean(dt,1);
%dtstd=std(dt,1);
%y=(dt)./repmat(dtstd,g,1);
y=dt;
tau=0.8;
delta0=0;


%--------MCMC parameters
nsim = 200;  
ncur = 1;     
nrun = 100-1;
burn = 200;

%--------initialize parameters
K=5;
St=unidrnd(K,n,1);
Stall=St;
gammatj=ones(g,n);
gammat=ones(n,1);
alphal=zeros(K-1,1);
mu=1;
mumu=0;
tautau=1;
atau=0.5;btau=0.5;
%taul=gamrnd(atau,1/btau,K,1);
taul=[0.4 1 1 1 1];
thetal=[-5 -2  0.1 1 2];
mu0=0;
tau0=1;
a0=0.5;
b0=0.5;
kj=0.5*ones(K,1);
k=0.5;

%Stout=zeros(nsim,g,n,"single");
Stout=zeros(nsim,n,"single");
alphalout=zeros(nsim,K-1,'single');
thetalout=zeros(nsim,K,'single');
taulout=zeros(nsim,K,'single');

%--------sampling
for gt=1:nsim
   
    %Update augmented variable Z_tl and weight alpha_l
    Ztl=zeros(n,K);
    %n1=size(gammat(gammat==1),1);
    for i = 1:n
        if Stall(i)<K
            for l = 1:Stall(i)
                v=1;
                m=alphal(l);
                if l<Stall(i)
                    pd = makedist('Normal',m,sqrt(v));
                    t = truncate(pd,-inf,0);
                    Ztl(i,l)=random(t,1,1);
                elseif l==Stall(i)
                    pd = makedist('Normal',m,sqrt(v));
                    t = truncate(pd,0,inf);
                    Ztl(i,l)=random(t,1,1);
                end
            end
        elseif Stall(i)==K
            for l = 1:K-1
                v=1;
                m=alphal(l);
                pd = makedist('Normal',m,sqrt(v));
                t = truncate(pd,-inf,0);
                Ztl(i,l)=random(t,1,1);
            end
        end
    end
    
    %Ztl(gammat==0,:)=0;
    for l=1:K-1
    v=inv(1+size(Stall(ge(Stall,l)==1),1));
    m=v*(sum(Ztl(ge(Stall,l)==1,l))+mu);
    alphal(l)=normrnd(m,sqrt(v),1,1);
    end
    
    
    %Update mu mu~N(mumu,tautau)
    tautauhat= (K)+tautau;
    mumuhat=(sum(alphal)+mumu*tautau)/tautauhat;
    mu=normrnd(mumuhat,sqrt(1/tautauhat));
    
    
    %Update St for t=1,...,n, given gamma(j,t)=1 with
    %omega_l=phi(alpha_l)prod(1-phi(r))
    phxi=zeros(n,K);
    %n1=size(gammat(gammat==1),1);
            for i=1:n
                vhx=ones(K-1,1);
                phx=ones(K,1);
                for h=1:K-1
                    vhx(h,1)=normcdf(alphal(1),0,1);
                    if h==1
                        phx(h)=vhx(h);
                    elseif h>1
                        phx(h)=vhx(h)*prod(1-vhx(1:h-1,1));
                    end
                end
                phx(K)=prod(1-vhx);phxi(i,:)=phx';
                
               if max(sum(cluster(:,i)==1),sum(cluster(:,i)==3))>11
                    if sum(cluster(:,i)==1)>= sum(cluster(:,i)==3)
                        state=1;
                    else
                        state=3;
                    end
                    sumL=zeros(K,1);
                    for l= 1:K
                    %logl=sum(log(normpdf(y(:,i),thetal(l),1/sqrt(taul(l)))+realmin).*gammatj(:,i));
                         logl=sum(log(normpdf(y(cluster(:,i)==state,i),thetal(l),1/sqrt(taul(l)))+realmin));
                         sumL(l)=sum(logl);
                     end
                        
                     phx1=exp(log(phx+realmin)+sumL)+realmin;
                     phx12=phx1/sum(phx1);
                     St(i)=randsample(K,1,true,phx12);
                else
                     sumL=zeros(K,1);
                     for l= 1:K
                     %logl=sum(log(normpdf(y(:,i),thetal(l),1/sqrt(taul(l)))+realmin));
                     %sumL(l)=sum(logl);
                     sumL(l)=sum(log(normpdf(y(:,i),thetal(l),1/sqrt(taul(l)))+realmin));
                     end
                    
                     phx1=exp(log(phx+realmin)+sumL)+realmin;
                     phx12=phx1/sum(phx1);
                     St(i)=randsample(K,1,true,phx12);
               end
            end
            Stall=St;
            St(gammat==0)=0;
            St(max(sum(cluster(:,i)==1),sum(cluster(:,i)==3))<11)=0;
            
            
     
            
     mubj=zeros(K,1);
     taubj=zeros(K,1);
     for l = 1:K
         if l<3
             state=1;
         elseif l==3 
             state=2;
         else
             state=3;
         end
         yl=y(:,(St==l)); 
         clusterl=cluster(:,(St==l));
         nl=size(yl,2);
         size1=zeros(nl,1);
         yi=zeros(nl,1);
         for i =1:nl
                yi(i)=sum(yl((clusterl(:,i)==state),i));
                size1(i)=length(yl((clusterl(:,i)==state),i));
         end    
         n2=sum(size1);
         m=(mu0*tau0+sum(yi)*taul(l))/(tau0+n2*taul(l));
         v=tau0+n2*taul(l);
         mubj(l)=m;
         taubj(l)=v;
         thetal(l)=normrnd(m,1/sqrt(v),1,1);
     end
 

    
     %Update component precision taul gamma(atau, btau)
     for l = 1:K
         if l<3
             state=1;
         elseif l==3 
             state=2;
         else
             state=3;
         end
         yl=y(:,(St==l)); 
         clusterl=cluster(:,(St==l));
         nl=size(yl,2);
         size1=zeros(nl,1);
         rse=zeros(n,1);
         for i =1:nl
             rse(i)=sum((yl((clusterl(:,i)==state),i)-thetal(l)).^2);
             size1(i)=length(yl((clusterl(:,i)==state),i));
         end
       aa=atau+0.5*sum(size1);
       bb=btau+0.5*sum(rse);
       taul(l)=gamrnd(aa,1/bb);
     end       
            
     %update gammatj
    for i = 1:n
         %ystar=y(:,i)-repmat(thetal(Stall(i)),g,1);
         ystar=y(:,i)-repmat(thetal(Stall(i)),g,1);
         %1:14
         if St(i)<3
             state=1;
         elseif St(i)==3
             state=2;
         else
             state=3;
         end
         g1=length(ystar(cluster(:,i)==state));
         mmu=(mu0*tau0+sum(ystar(cluster(:,i)==state))*taul(Stall(i)))/(tau0+g1*taul(Stall(i)));
         mv=tau0+g1*taul(Stall(i));
         at=log(k)+sum(log(normpdf(ystar(cluster(:,i)==state),0,1/sqrt(taul(Stall(i))))))+log(normpdf(0,mu0,1/sqrt(tau0))+realmin)...
             -log(normpdf(0,mmu,1/sqrt(mv))+realmin);
         %at=log(k)+sum(log(normpdf(ystar,0,1/sqrt(taul(Stall(i))))))+log(normpdf(0,mubj(Stall(i)),1/sqrt(taubj(Stall(i))))+realmin)...
             %-log(normpdf(0,mmu,1/sqrt(mv))+realmin);
         bt=log(1-k)+sum(log(normpdf(y((cluster(:,i)==state),i),delta0,1/sqrt(tau))+realmin));
         gammatj((cluster(:,i)==state),i)=binornd(1,1/(1+exp(bt-at)),g1,1);
         
         g2=length(ystar(cluster(:,i)~=state));
         gammatj((cluster(:,i)~=state),i)=zeros(g2,1);
     end
        
      for i =1:n
         sum_gamma=mean(gammatj(:,i));
         if sum_gamma>0.10
             gammat(i)=1;
         elseif sum_gamma<=0.10
             gammat(i)=0;
         end
     end
     %Updata variable selection kj Bernoulli-Beta(a0,b0)
     

     
     
     
     %osum=(sum((gammat==1))==0);
     %output
     gt
     thetal
     taul
     kj
     alphalout(gt,:)=alphal;
     Stout(gt,:,:)=St;
     thetalout(gt,:)=thetal;
     taulout(gt,:)=taul;
     
            
    
end

writematrix(Stout, 'D:\Desktop\stoutchr22.txt');
writematrix(thetalout, 'D:\Desktop\thetaloutchr22.txt');
writematrix(taulout, 'D:\Desktop\tauloutchr22.txt');

