#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <cstring>
using namespace std;  

#define N 40
#define M 40
#define a(i,j) a[(i)*(N+1)+j]
#define b(i,j) b[(i)*(N+1)+j]
#define w(i,j) w[(i)*(N+1)+j]
#define w1(i,j) w1[(i)*(N+1)+j]
#define F(i,j) F[(i)*(N+1)+j]
#define r(i,j) r[(i)*(N+1)+j]

int subdiv(int K, int m, int n);

static int *ix1, *ix2, *jy1, *jy2;

int main(int argc,char *argv[])
{
    double *a = new double [(M+1)*(N+1)]();
    double *b = new double [(M+1)*(N+1)]();
    double *F = new double [(M+1)*(N+1)]();
    double *w = new double [(M+1)*(N+1)]();
    double *r = new double [(M+1)*(N+1)]();
	double *w1;
    double x1=-1, x2=1, y1=-0.5, y2=0.5;   // границы фиктивной области
    double h1=(x2-x1)/M, h2=(y2-y1)/N;     // шаги сетки по x и по y
    double eps=h1*h2;

    int K=subdiv(9, M, N);

    //cout << K << endl;
    //for (int k=0; k<K; k++)
    //{
    //    cout << ix1[k] << ' ';
    //    cout << ix2[k] << ' ';
    //    cout << jy1[k] << ' ';
    //    cout << jy2[k] << endl;
    //}
	
    int nproc, pid;

      /* Инициализируем библиотеку */
    MPI_Init( &argc, &argv );

      /* Узнаем количество задач в запущенном приложении */
    MPI_Comm_size( MPI_COMM_WORLD, &nproc );

      /* ... и свой собственный номер: от 0 до (size-1) */
    MPI_Comm_rank( MPI_COMM_WORLD, &pid );

	if (pid==0)
	{
		w1 = new double [(M+1)*(N+1)]();
	}
    int nthr=atoi(argv[1]);
    omp_set_num_threads(nthr);
    double *dwthr = new double [nthr]();
    
	double starttime, endtime;
    starttime = MPI_Wtime();
    double time = omp_get_wtime();
    nthr=omp_get_max_threads();
	cout << "number of threads=" << nthr << endl;
	
    // вычисление коэффициентов a_ij
    #pragma omp parallel for shared(a,x1,h1,y1,h2,eps)  //private(i,j,x,el,ya,yb,l)
    for (int i=1; i<=M; i++)
    {
        double x=x1+(i-0.5)*h1;
        double el=0.5*sqrt(1-x*x);
        for (int j=1; j<=N; j++)
        {
            double ya=y1+(j-0.5)*h2;
            double yb=ya+h2;
            if (yb>=-el) yb=min(el,yb);
            else   // отрезок [ya,yb] ниже реальной области (эллипса)
            {
               a(i,j)=1.0/eps;   
               continue;
            }
            if (ya<=el) ya=max(-el,ya);
            else    // отрезок [ya,yb] выше реальной области (эллипса)
            {
               a(i,j)=1.0/eps;
               continue;
            }
            double l=(yb-ya)/h2;
            a(i,j) = l+(1-l)/eps;
        }
    }

    // вычисление коэффициентов b_ij
    #pragma omp parallel for shared(b,x1,h1,y1,h2,eps)
    for (int j=1; j<=N; j++)
    {
        double y=y1+(j-0.5)*h2;
        double el=sqrt(1-4*y*y);
        for (int i=1; i<=M; i++)
        {
            double xa=x1+(i-0.5)*h1;
            double xb=xa+h1;
            if (xb>=-el) xb=min(el,xb);
            else   // отрезок [xa,xb] левее реальной области (эллипса)
            {
               b(i,j)=1.0/eps;   
               continue;
            }
            if (xa<=el) xa=max(-el,xa);
            else    // отрезок [xa,xb] правее реальной области (эллипса)
            {
               b(i,j)=1.0/eps;
               continue;
            }
            double l=(xb-xa)/h1;
            b(i,j) = l+(1-l)/eps;
        }
    }

    // вычисление правой части F_ij
    #pragma omp parallel for shared(F,x1,h1,y1,h2)
    for (int i=1; i<M; i++)
    {
        double xa=x1+(i-0.5)*h1;
        double xb=xa+h1;
        for (int j=1; j<N; j++)
        {
            double ya=y1+(j-0.5)*h2;
            double yb=ya+h2;
            int n=100;
            double dx=(xb-xa)/n;
            double S=0;
            for (int k=0; k<n; k++)
            {
                double x=xa+(k+0.5)*dx;
                double el=0.5*sqrt(1-x*x);
                S += dx*max(0.0,min(yb,el)-max(ya,-el));
            }
            F(i,j)=S/h1/h2;
        }
    }

    // итерации Якоби
    int it=0;
    double dwmax=0;
    for (it=0;it<10000; it++)
    {
        for (int i=0; i<nthr; i++) dwthr[i]=0;
		
		for (int k=pid; k<K; k +=nproc)
		{
			// вычисление вектора
			// r_ij = F_ij - A_{i+1,j} w_{i+1,j} - A_{i-1,j} w_{i-1,j} - A_{i,j+1} w_{i,j+1} - A_{i,j-1} w_{i,j-1}
			#pragma omp parallel for shared(a,b,F,w,r,h1,h2)
			for (int i=ix1[k]; i<ix2[k]; i++)
			{
				for (int j=jy1[k]; j<jy2[k]; j++)
				{
					r(i,j) =  F(i,j);
					r(i,j) += (a(i+1,j)*w(i+1,j) + a(i,j)*w(i-1,j))/h1/h1;
					r(i,j) += (b(i,j+1)*w(i,j+1) + b(i,j)*w(i,j-1))/h2/h2;
				}
			}

			// изменение текущего приближения решения w
			// w_ij = r_ij /A_ij
			#pragma omp parallel for shared(a,b,r,w,h1,h2,dwthr)
			for (int i=ix1[k]; i<ix2[k]; i++)
			{
				int id = omp_get_thread_num();
				for (int j=jy1[k]; j<jy2[k]; j++)
				{
					double wo=w(i,j);
					w(i,j) = r(i,j)/((a(i+1,j) + a(i,j))/h1/h1 + (b(i,j+1) + b(i,j))/h2/h2);
					double dw=abs(w(i,j)-wo);
					// максимум изменения w для каждого треда
					if ( dwthr[id] < dw )
					{
						dwthr[id] = dw;
					}	
				}
			}
			
		}
		
		dwmax=0;
        for (int i=0; i<nthr; i++)
		{
			if (dwmax<dwthr[i])
			{
				dwmax=dwthr[i];
			}
			//dwmax=max(dwmax,dwthr[i]); // максимум изменения w по потокам
		}
		MPI_Barrier( MPI_COMM_WORLD );	

		// объединить в управляющем процессе решения от остальных процессов и 
        if (pid!=0)
		{
            MPI_Send(w, (M+1)*(N+1), MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD);
		}
        else
		{
			for (int id=1; id<nproc; id++)
			{
				MPI_Recv(w1, (M+1)*(N+1), MPI_DOUBLE_PRECISION, id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (int k=id; k<K; k +=nproc)
				//#pragma omp parallel for
				for (int i=ix1[k]; i<ix2[k]; i++)
				for (int j=jy1[k]; j<jy2[k]; j++)
				{
					w(i,j) = w1(i,j);
				}
			}
        }

		// передать объединенное решение из управляющего процесса в остальные процессы
        if (pid==0)
		{
			// send updated solution to all processes
			for (int id=1; id<nproc; id++)
				MPI_Send(w, (M+1)*(N+1), MPI_DOUBLE_PRECISION, id, 0, MPI_COMM_WORLD);
		}
        else
		{
			MPI_Recv(w, (M+1)*(N+1), MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
		
		// максимальное изменение решения по процессам
        if (pid!=0)
		{
            MPI_Send(&dwmax, 1, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD);
		}
        else
		{
			for (int id=1; id<nproc; id++)
			{
				double dwmax1=0;
				MPI_Recv(&dwmax1, 1, MPI_DOUBLE_PRECISION, id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (dwmax < dwmax1) dwmax=dwmax1;
			}
			if (it%1000==0) cout << "It = " << it << ", dw = " << dwmax << endl;
        }

		// разослать глобальное изменение решения по процессам для использования в критерии окончания итераций
        if (pid==0)
		{
			for (int id=1; id<nproc; id++)
            MPI_Send(&dwmax, 1, MPI_DOUBLE_PRECISION, id, 0, MPI_COMM_WORLD);
		}
        else
		{
			MPI_Recv(&dwmax, 1, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

		if (dwmax<1e-9) break;    // выход из цикла итераций Якоби при достижении точности

    }
    

    // вывод решения в файл в основном процессе
	if (pid==0)
	{
		ofstream rslt;
		rslt.open ("result.txt");
		for (int j=0; j<=N; j++)
		{
			rslt << w(0,j);
			for (int i=1; i<=M; i++)
				rslt << ",\t" << w(i,j);
			rslt << endl;
		}
		rslt.close();
	}
	
    if (pid==0)
	{
		cout << "MY TIME OF PROG = " << omp_get_wtime() - time << endl;
		cout << "It = " << it << ", dw = " << dwmax << endl;
		endtime   = MPI_Wtime();
		printf("Mpi Time %f seconds\n",endtime-starttime);
	}
	
    delete[] a;
    delete[] b;
    delete[] F;
    delete[] w;
    delete[] r;
    /* Все задачи завершают выполнение */
    MPI_Finalize();
    return 0;
}

// деление на K подобластей
int subdiv(int K, int m0, int n0)
{
	int m=m0-1;
	int n=m0-1;
    int nx0=round(0.5*sqrt((2.0*K*m)/n)+0.5*sqrt((K*m)/(2.0*m)));
    int ny0=round((1.0*K)/nx0);
    if (nx0<ny0 && m>=n) swap(nx0,ny0);
    if (nx0>ny0 && m<n) swap(nx0,ny0);
    ix1=new int [nx0*ny0]();
    ix2=new int [nx0*ny0]();
    jy1=new int [nx0*ny0]();
    jy2=new int [nx0*ny0]();
    int ix=1, jy=1;
    int K1=0;
    for (int i=0; i<nx0; i++)
    {
        int ixn = ix + (m0-ix)/(nx0-i);
        jy=1;
        for (int j=0; j<ny0; j++)
        {
            int jyn = jy + (n0-jy)/(ny0-j);
            ix1[K1]=ix; 
            ix2[K1]=ixn;
            jy1[K1]=jy; 
            jy2[K1]=jyn;
            K1++;
            jy=jyn;
        }
        ix=ixn;
    }
    return K1;
}
