import numpy as np, math

def rastrigin(x):
    n=len(x);return 10*n+np.sum(x**2-10*np.cos(2*math.pi*x))

def levy_flight(beta,dim):
    num=math.gamma(1+beta)*math.sin(math.pi*beta/2)
    den=math.gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.normal(0,sigma,dim);v=np.random.normal(0,1,dim)
    return u/np.abs(v)**(1/beta)

def cuckoo_search(f,dim=2,lb=-5,ub=5,n=10,pa=0.25,alpha=0.01,beta=1.5,iters=5):
    nests=lb+(ub-lb)*np.random.rand(n,dim)
    fit=np.array([f(x) for x in nests])
    best=nests[np.argmin(fit)];best_fit=min(fit)
    for t in range(iters):
        for i in range(n):
            step=levy_flight(beta,dim)
            new=nests[i]+alpha*step*(nests[i]-best)
            new=np.clip(new,lb,ub);fnew=f(new);j=np.random.randint(n)
            if fnew<fit[j]:
                nests[j],fit[j]=new,fnew
                if fnew<best_fit:best,best_fit=new,fnew
        K=int(pa*n)
        worst=np.argsort(fit)[-K:]
        for idx in worst:
            nests[idx]=lb+(ub-lb)*np.random.rand(dim)
            fit[idx]=f(nests[idx])
            if fit[idx]<best_fit:best,best_fit=nests[idx],fit[idx]
        print("Iter",t+1,"Best fitness",best_fit)
    return best,best_fit

best,best_fit=cuckoo_search(rastrigin)
print("Best:",best,"Fitness:",best_fit)
