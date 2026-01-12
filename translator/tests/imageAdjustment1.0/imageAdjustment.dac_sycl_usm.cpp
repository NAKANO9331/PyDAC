#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// 定义图像类型，假设每个像素由 RGB 三个值组成
struct Pixel {
    int r, g, b;
    // 重载 << 操作符，使 Pixel 类型的对象可以被输出
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};




// 色彩调整操作：增加红色分量


// 亮度增强操作：增加每个像素的 RGB 分量




// 打印图像的前几个像素，作为调试
void print_image(const std::vector<std::vector<Pixel>>& image, int num_rows = 5, int num_cols = 5) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << "(" << image[i][j].r << "," << image[i][j].g << "," << image[i][j].b << ") ";
        }
        std::cout << std::endl;
    }
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void image_1(Pixel* image_tensor,Pixel* image_tensor2,sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor2_acc) 
{
    image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);
}


// 生成函数调用
void imageAdjustment_image_1(const dacpp::Matrix<Pixel> & image_tensor, dacpp::Matrix<Pixel> & image_tensor2) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_image_tensor;
    info_image_tensor.dim = image_tensor.getDim();
    for(int i = 0; i < info_image_tensor.dim; i++) info_image_tensor.dimLength.push_back(image_tensor.getShape(i));
	
    // 数据信息初始化
    DataInfo info_image_tensor2;
    info_image_tensor2.dim = image_tensor2.getDim();
    for(int i = 0; i < info_image_tensor2.dim; i++) info_image_tensor2.dimLength.push_back(image_tensor2.getShape(i));
	
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_image_tensor));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_image_tensor));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops image_tensor_Ops;
    
    idx1.setDimId(0);
    image_tensor_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops image_tensor2_Ops;
    
    idx1.setDimId(0);
    image_tensor2_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor2_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int image_tensor_Size = para_gene_tool.init_device_memory_size(info_image_tensor,image_tensor_Ops);

    //生成设备内存分配大小
    int image_tensor2_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_image_tensor2);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(image_tensor_Ops,image_tensor_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,image_tensor2_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(image_tensor_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	    
	
    // 设备内存分配
    Pixel *d_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    // 设备内存分配
    Pixel *d_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    // 数据移动
	Pixel* h_image_tensor = (Pixel*)malloc(image_tensor_Size*sizeof(Pixel));
	image_tensor.tensor2Array(h_image_tensor);
    q.memcpy(d_image_tensor,h_image_tensor,image_tensor_Size*sizeof(Pixel)).wait();

    // 数据移动
	Pixel* h_image_tensor2 = (Pixel*)malloc(image_tensor2_Size*sizeof(Pixel));
	// image_tensor2.tensor2Array(h_image_tensor2);
    q.memset(d_image_tensor2, 0, image_tensor2_Size*sizeof(Pixel)).wait();
    // 数据重组
    DataReconstructor<Pixel> image_tensor_tool;
    
    // 数据算子组初始化
    Dac_Ops image_tensor_ops;
    
    idx1.setDimId(0);
    image_tensor_ops.push_back(idx1);
    idx2.setDimId(1);
    image_tensor_ops.push_back(idx2);

    image_tensor_tool.init(info_image_tensor,image_tensor_ops,q);
	Pixel *r_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    image_tensor_tool.Reconstruct(r_image_tensor,d_image_tensor,q);
	std::vector<int> info_partition_image_tensor=para_gene_tool.init_partition_data_shape(info_image_tensor,image_tensor_ops);
    sycl::buffer<int> info_partition_image_tensor_buffer(info_partition_image_tensor.data(), sycl::range<1>(info_partition_image_tensor.size()));

    // 数据重组
    DataReconstructor<Pixel> image_tensor2_tool;
    
    // 数据算子组初始化
    Dac_Ops image_tensor2_ops;
    
    idx1.setDimId(0);
    image_tensor2_ops.push_back(idx1);
    idx2.setDimId(1);
    image_tensor2_ops.push_back(idx2);

    image_tensor2_tool.init(info_image_tensor2,image_tensor2_ops,q);
	Pixel *r_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    image_tensor2_tool.Reconstruct(r_image_tensor2,d_image_tensor2,q);
	std::vector<int> info_partition_image_tensor2=para_gene_tool.init_partition_data_shape(info_image_tensor2,image_tensor2_ops);
    sycl::buffer<int> info_partition_image_tensor2_buffer(info_partition_image_tensor2.data(), sycl::range<1>(info_partition_image_tensor2.size()));

	
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
        
        auto info_partition_image_tensor_accessor = info_partition_image_tensor_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_image_tensor2_accessor = info_partition_image_tensor2_buffer.get_access<sycl::access::mode::read_write>(h);

		h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            image_1(r_image_tensor+(idx1_*SplitLength[0][0]+idx2_*SplitLength[0][1]),r_image_tensor2+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),info_partition_image_tensor_accessor,info_partition_image_tensor2_accessor);
        });
    }).wait();
    

	
    // 归并结果返回
    image_tensor2_tool.UpdateData(r_image_tensor2,d_image_tensor2,q,image_tensor2_Size);
	q.memcpy(h_image_tensor2,d_image_tensor2, image_tensor2_Size*sizeof(Pixel)).wait();
	image_tensor2.array2Tensor(h_image_tensor2);

	

    // 内存释放
    
    sycl::free(d_image_tensor, q);
    sycl::free(d_image_tensor2, q);
}

void image_2(Pixel* image_tensor2,Pixel* image_tensor3,sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor2_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor3_acc) 
{
    int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);
}


// 生成函数调用
void imageAdjustment_image_2(const dacpp::Matrix<Pixel> & image_tensor, dacpp::Matrix<Pixel> & image_tensor2) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_image_tensor;
    info_image_tensor.dim = image_tensor.getDim();
    for(int i = 0; i < info_image_tensor.dim; i++) info_image_tensor.dimLength.push_back(image_tensor.getShape(i));
	
    // 数据信息初始化
    DataInfo info_image_tensor2;
    info_image_tensor2.dim = image_tensor2.getDim();
    for(int i = 0; i < info_image_tensor2.dim; i++) info_image_tensor2.dimLength.push_back(image_tensor2.getShape(i));
	
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_image_tensor));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_image_tensor));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops image_tensor_Ops;
    
    idx1.setDimId(0);
    image_tensor_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops image_tensor2_Ops;
    
    idx1.setDimId(0);
    image_tensor2_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor2_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int image_tensor_Size = para_gene_tool.init_device_memory_size(info_image_tensor,image_tensor_Ops);

    //生成设备内存分配大小
    int image_tensor2_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_image_tensor2);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(image_tensor_Ops,image_tensor_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,image_tensor2_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(image_tensor_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	    
	
    // 设备内存分配
    Pixel *d_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    // 设备内存分配
    Pixel *d_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    // 数据移动
	Pixel* h_image_tensor = (Pixel*)malloc(image_tensor_Size*sizeof(Pixel));
	image_tensor.tensor2Array(h_image_tensor);
    q.memcpy(d_image_tensor,h_image_tensor,image_tensor_Size*sizeof(Pixel)).wait();

    // 数据移动
	Pixel* h_image_tensor2 = (Pixel*)malloc(image_tensor2_Size*sizeof(Pixel));
	// image_tensor2.tensor2Array(h_image_tensor2);
    q.memset(d_image_tensor2, 0, image_tensor2_Size*sizeof(Pixel)).wait();
    // 数据重组
    DataReconstructor<Pixel> image_tensor_tool;
    
    // 数据算子组初始化
    Dac_Ops image_tensor_ops;
    
    idx1.setDimId(0);
    image_tensor_ops.push_back(idx1);
    idx2.setDimId(1);
    image_tensor_ops.push_back(idx2);

    image_tensor_tool.init(info_image_tensor,image_tensor_ops,q);
	Pixel *r_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    image_tensor_tool.Reconstruct(r_image_tensor,d_image_tensor,q);
	std::vector<int> info_partition_image_tensor=para_gene_tool.init_partition_data_shape(info_image_tensor,image_tensor_ops);
    sycl::buffer<int> info_partition_image_tensor_buffer(info_partition_image_tensor.data(), sycl::range<1>(info_partition_image_tensor.size()));

    // 数据重组
    DataReconstructor<Pixel> image_tensor2_tool;
    
    // 数据算子组初始化
    Dac_Ops image_tensor2_ops;
    
    idx1.setDimId(0);
    image_tensor2_ops.push_back(idx1);
    idx2.setDimId(1);
    image_tensor2_ops.push_back(idx2);

    image_tensor2_tool.init(info_image_tensor2,image_tensor2_ops,q);
	Pixel *r_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    image_tensor2_tool.Reconstruct(r_image_tensor2,d_image_tensor2,q);
	std::vector<int> info_partition_image_tensor2=para_gene_tool.init_partition_data_shape(info_image_tensor2,image_tensor2_ops);
    sycl::buffer<int> info_partition_image_tensor2_buffer(info_partition_image_tensor2.data(), sycl::range<1>(info_partition_image_tensor2.size()));

	
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
        
        auto info_partition_image_tensor_accessor = info_partition_image_tensor_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_image_tensor2_accessor = info_partition_image_tensor2_buffer.get_access<sycl::access::mode::read_write>(h);

		h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            image_2(r_image_tensor+(idx1_*SplitLength[0][0]+idx2_*SplitLength[0][1]),r_image_tensor2+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),info_partition_image_tensor_accessor,info_partition_image_tensor2_accessor);
        });
    }).wait();
    

	
    // 归并结果返回
    image_tensor2_tool.UpdateData(r_image_tensor2,d_image_tensor2,q,image_tensor2_Size);
	q.memcpy(h_image_tensor2,d_image_tensor2, image_tensor2_Size*sizeof(Pixel)).wait();
	image_tensor2.array2Tensor(h_image_tensor2);

	

    // 内存释放
    
    sycl::free(d_image_tensor, q);
    sycl::free(d_image_tensor2, q);
}

int main() {
    // 初始化一个简单的图像（10x10），所有像素值初始化为(100, 100, 100)

    int width, height;
    std::cout << "Enter width: ";
    std::cin >> width;  // 错误：无法修改const变量

    std::cout << "Enter height: ";
    std::cin >> height;  // 错误：无法修改const变量
    std::vector<Pixel> image(height*width, {100, 100, 100});
    std::vector<Pixel> image2(height*width, {100, 100, 100});
    //std::vector<std::vector<Pixel>> image2(height, std::vector<Pixel>(width, {100, 100, 100}));

    // 打印初始图像
    std::cout << "Original Image:" << std::endl;
    //print_image(image);

    dacpp::Matrix<Pixel> image_tensor({height, width}, image);
    dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);

    // 执行色彩调整操作
    imageAdjustment_image_1(image_tensor, image_tensor2);
    std::cout << "\nImage After Color Adjustment:" << std::endl;

    // From updated image_tensor2, get data to create image3
    // First get the shape and size of image_tensor2
    int tensor2_height = image_tensor2.getShape(0);
    int tensor2_width = image_tensor2.getShape(1);
    std::vector<Pixel> image3;
    image_tensor2.tensor2Array(image3);
    // Use the actual shape of image_tensor2 to create image_tensor3
    dacpp::Tensor<Pixel, 2> image_tensor3({tensor2_height, tensor2_width}, image3);


    // 执行亮度增强操作
    imageAdjustment_image_2(image_tensor2, image_tensor3);
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();

    return 0;
}
