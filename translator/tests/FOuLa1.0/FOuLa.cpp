#include <cmath>
#include <stdlib.h>
#include <stdio.h>
 
int main(int argc, char *argv[])
{
        int n,m,i,j,k,number;
        double a,h,tau,r;
        double *x,*t,**u;
        double phi(double x);
        double alpha(double t);
        double beta(double t);
        double f(double x, double t);
        double exact(double x, double t);
 
        n=100;  //时间域n等分
        m=5;    //空间域m等分
        a=1.0;
        h=1.0/m;  //空间步长
        tau=1.0/n;  //时间步长
        r=a*tau/(h*h);  //网比
        printf("r=%.4f.\n",r);
 
        x=(double*)malloc(sizeof(double)*(m+1));
        for(i=0;i<=m;i++)
                x[i]=i*h;
 
        t=(double*)malloc(sizeof(double)*(n+1));
        for(i=0;i<=n;i++)
                t[i]=i*tau;
 
        u=(double**)malloc(sizeof(double*)*(m+1));
        for(i=0;i<=m;i++)
                u[i]=(double*)malloc(sizeof(double)*(n+1));
        for(i=0;i<=m;i++)
                u[i][0]=phi(x[i]);
        for(i=1;i<=n;i++)
        {
                u[0][i]=alpha(t[i]);
                u[m][i]=beta(t[i]);
        }
 
        for(k=0;k<n;k++)
        {
                for(i=1;i<m;i++)
                {
                        u[i][k+1]=r*u[i-1][k]+(1-2*r)*u[i][k]+r*u[i+1][k]+tau*f(x[i],t[k]);
                }
        }
 
        j=int(0.2/tau);
        number=int(0.4/h);
        for(k=j;k<=n;k=k+j)
        {
                printf("(x,t)=(%.1f,%.1f), y=%f, exact=%f, err=%.4e.\n",x[number],t[k],u[number][k],exact(x[number],t[k]),fabs(u[number][k]-exact(x[number],t[k])));
        }
 
        free(x);free(t);
        for(i=0;i<=m;i++)
                free(u[i]);
        free(u);
 
        return 0;
}
 
 
double phi(double x)
{
        return x*x*x+x;
}
 
double alpha(double t)
{
        return 0.0;
}
double beta(double t)
{
        return 1.0+exp(t);
}
double f(double x, double t)
{
        return x*exp(t)-6*x;
}
double exact(double x, double t)
{
        return x*(x*x+exp(t));
}