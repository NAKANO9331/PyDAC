#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <any>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <queue>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

double phi(double x) { return x*x*x+x; }

double alpha(double t) { return 0.0; }

double beta(double t) { return 1.0+exp(t); }

double f(double x, double t) { return x*exp(t)-6*x; }

double exact(double x, double t) { return x*(x*x+exp(t)); }

//同样的问题，划分时，一个待计算数据和三个计算数据，一共四个数据要划分到一起




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void pde(double* u_kin,double* u_kout,double* r,sycl::accessor<int, 1, sycl::access::mode::read_write> info_u_kin_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_u_kout_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_r_acc) 
{
    u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}


// 生成函数调用
void PDE_pde(const dacpp::Vector<double> & u_kin, dacpp::Vector<double> & u_kout, const dacpp::Vector<double> & r) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_u_kin;
    info_u_kin.dim = u_kin.getDim();
    for(int i = 0; i < info_u_kin.dim; i++) info_u_kin.dimLength.push_back(u_kin.getShape(i));
	
    // 数据信息初始化
    DataInfo info_u_kout;
    info_u_kout.dim = u_kout.getDim();
    for(int i = 0; i < info_u_kout.dim; i++) info_u_kout.dimLength.push_back(u_kout.getShape(i));
	
    // 数据信息初始化
    DataInfo info_r;
    info_r.dim = r.getDim();
    for(int i = 0; i < info_r.dim; i++) info_r.dimLength.push_back(r.getShape(i));
	
    // 规则分区算子初始化
    RegularSlice s = RegularSlice("s", 3, 1);
    s.setDimId(0);
    s.SetSplitSize(para_gene_tool.init_operetor_splitnumber(s,info_u_kin));

    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_u_kout));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops u_kin_Ops;
    
    s.setDimId(0);
    u_kin_Ops.push_back(s);


    // 算子组初始化
    Dac_Ops u_kout_Ops;
    
    i.setDimId(0);
    u_kout_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops r_Ops;
    

    // 算子组初始化
    Dac_Ops In_Ops;
    
    s.setDimId(0);
    In_Ops.push_back(s);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    i.setDimId(0);
    Reduction_Ops.push_back(i);


	
    //生成设备内存分配大小
    int u_kin_Size = para_gene_tool.init_device_memory_size(info_u_kin,u_kin_Ops);

    //生成设备内存分配大小
    int u_kout_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_u_kout);

    //生成设备内存分配大小
    int r_Size = para_gene_tool.init_device_memory_size(info_r,r_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(u_kin_Ops,u_kin_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,u_kout_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(r_Ops,r_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(u_kin_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(r_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[3][1] = {0};
    para_gene_tool.init_split_length_martix(3,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	    
	
    // 设备内存分配
    double *d_u_kin=malloc_device<double>(u_kin_Size,q);
    // 设备内存分配
    double *d_u_kout=malloc_device<double>(u_kout_Size,q);
    // 设备内存分配
    double *d_r=malloc_device<double>(r_Size,q);
    // 数据移动
	double* h_u_kin = (double*)malloc(u_kin_Size*sizeof(double));
	u_kin.tensor2Array(h_u_kin);
    q.memcpy(d_u_kin,h_u_kin,u_kin_Size*sizeof(double)).wait();

    // 数据移动
	double* h_u_kout = (double*)malloc(u_kout_Size*sizeof(double));
	// u_kout.tensor2Array(h_u_kout);
    q.memset(d_u_kout, 0, u_kout_Size*sizeof(double)).wait();
    // 数据移动
	double* h_r = (double*)malloc(r_Size*sizeof(double));
	r.tensor2Array(h_r);
    q.memcpy(d_r,h_r,r_Size*sizeof(double)).wait();

    // 数据重组
    DataReconstructor<double> u_kin_tool;
    
    // 数据算子组初始化
    Dac_Ops u_kin_ops;
    
    s.setDimId(0);
    u_kin_ops.push_back(s);

    u_kin_tool.init(info_u_kin,u_kin_ops,q);
	double *r_u_kin=malloc_device<double>(u_kin_Size,q);
    u_kin_tool.Reconstruct(r_u_kin,d_u_kin,q);
	std::vector<int> info_partition_u_kin=para_gene_tool.init_partition_data_shape(info_u_kin,u_kin_ops);
    sycl::buffer<int> info_partition_u_kin_buffer(info_partition_u_kin.data(), sycl::range<1>(info_partition_u_kin.size()));

    // 数据重组
    DataReconstructor<double> u_kout_tool;
    
    // 数据算子组初始化
    Dac_Ops u_kout_ops;
    
    i.setDimId(0);
    u_kout_ops.push_back(i);

    u_kout_tool.init(info_u_kout,u_kout_ops,q);
	double *r_u_kout=malloc_device<double>(u_kout_Size,q);
    u_kout_tool.Reconstruct(r_u_kout,d_u_kout,q);
	std::vector<int> info_partition_u_kout=para_gene_tool.init_partition_data_shape(info_u_kout,u_kout_ops);
    sycl::buffer<int> info_partition_u_kout_buffer(info_partition_u_kout.data(), sycl::range<1>(info_partition_u_kout.size()));

    // 数据重组
    DataReconstructor<double> r_tool;
    
    // 数据算子组初始化
    Dac_Ops r_ops;
    

    r_tool.init(info_r,r_ops,q);
	double *r_r=malloc_device<double>(r_Size,q);
    r_tool.Reconstruct(r_r,d_r,q);
	std::vector<int> info_partition_r=para_gene_tool.init_partition_data_shape(info_r,r_ops);
    sycl::buffer<int> info_partition_r_buffer(info_partition_r.data(), sycl::range<1>(info_partition_r.size()));

	
    sycl::device device = q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16×16，但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_u_kin_accessor = info_partition_u_kin_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_u_kout_accessor = info_partition_u_kout_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_r_accessor = info_partition_r_buffer.get_access<sycl::access::mode::read_write>(h);

		h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            const auto s_=(item_id+(0))%s.split_size;
            // 嵌入计算
			
            pde(r_u_kin+(s_*SplitLength[0][0]),r_u_kout+(s_*SplitLength[1][0]),r_r,info_partition_u_kin_accessor,info_partition_u_kout_accessor,info_partition_r_accessor);
        });
    }).wait();
    

	
    // 归并结果返回
    u_kout_tool.UpdateData(r_u_kout,d_u_kout,q,u_kout_Size);
	q.memcpy(h_u_kout,d_u_kout, u_kout_Size*sizeof(double)).wait();
	u_kout.array2Tensor(h_u_kout);

	

    // 内存释放
    
    sycl::free(d_u_kin, q);
    sycl::free(d_u_kout, q);
    sycl::free(d_r, q);
}

int main() {
    int n = 100; //时间域n等分
    int m = 5; //空间域m等分
    double r = 0.25;
    double a = 1.0;
    double h = 1.0 / m; //空间步长
    double tau = 1.0 / n; //时间步长
    double *x,*t,**u;
    
    //r=a*tau/(h*h);  //网比
    //printf("r=%.4f.\n",r);
    
    x = (double*)malloc(sizeof(double)*(m+1));
    for (int i=0;i<=m;i++) {
        x[i]=i*h;
    }
    t = (double*)malloc(sizeof(double)*(n+1));
    for (int i = 0; i <= n; i++) {
        t[i]=i*tau;
    }
    u = (double**)malloc(sizeof(double*)*(m+1));
    for (int i=0;i<=m;i++) {
        u[i]=(double*)malloc(sizeof(double)*(n+1));
    }
    for (int i = 0; i <= m; i++)
        u[i][0]=phi(x[i]);
    for (int i = 1; i <= n; i++) {
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }
    
    // Flatten the 2D u array into a 1D vector for Tensor creation
    std::vector<double> u_flat;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            u_flat.push_back(static_cast<double>(u[i][j]));  // Cast if needed
        }
    }

    dacpp::Matrix<double> u_tensor({m+1, n+1}, u_flat);

    for (int k = 0; k < n; k++) {
        dacpp::Vector<double> middle_tensor = u_tensor[{1,m}][k+1];
        std::vector<double> r_data;
        r_data.push_back(r);
        dacpp::Vector<double> R(r_data);
        dacpp::Vector<double> u_test1 = u_tensor[{}][k];
        PDE_pde(u_test1, middle_tensor, R);
        
        //计算完毕后，替换第1到4个点
        for (int i = 1; i <= m-1; i++) {
            u_tensor[i][k+1] = middle_tensor[i-1];
        }

    }

    // 每个位置需要下，左下，右下，三个位置的元素，串行中从下往上，从左往右遍历计算
    // 那么每一行的元素计算是互不相关的，可以并行执行，所有的行从下往上串行执行
    u_tensor[1].print();
    // double* data = new double[6 * 101];
    // u_tensor.tensor2Array(data);

    // // 将一维数组转换为二维 vector
    // std::vector<std::vector<double>> vec2D;
    // vec2D.resize(6, std::vector<double>(101));

    // // 将一维数组的数据填充到二维数组中
    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 101; ++j) {
    //         vec2D[i][j] = data[i * 101 + j];
    //     }
    // }


    // int j = int(0.2 / tau);
    // int number = int(0.4 / h);
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f, err=%.3e.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]),std::fabs(vec2D[number][k]-exact(x[number],t[k])));
    // }
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]));
    // }


    return 0;
}