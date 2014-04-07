def hyp(x,theta):
    h=[]
    for i in range(len(x)):
        h.append(theta[0]+theta[1]*x[i])
    return h
 
def cost(h,y):
    return (1./(2*len(h)))*sum(map(lambda w,x:(w-x)**2,h,y))
 
def grad_descent(theta,h,y,x,alpha,n_iter):
    thetas=[]
    costs=[]
    for i in range(n_iter):
        temp_0=theta[0]-alpha*sum(map(lambda w,r:(w-r),h,y))*(1./len(x))
        temp_1=theta[1]-alpha*sum(map(lambda w,r,s:(w-r)*s,h,y,x))*(1./len(x))
        theta[0]=temp_0
        theta[1]=temp_1
        h=hyp(x,theta)
        j=cost(h,y)
        thetas.append((theta[0],theta[1]))
        costs.append(j)
    return thetas,costs
 
def linear_regression_1(x,y,alpha,n_iter):
    theta=[1.0,1.0]
    h=hyp(x,theta)
    thetas,costs=grad_descent(theta,h,y,x,alpha,n_iter)
    return thetas,costs
 
x=[5,3,0,4]
y=[4,4,1,3]
thetas,costs=linear_regression_1(x,y)