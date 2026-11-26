/* pybind11 的连接文件 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "HNSW.hpp"
namespace py = pybind11;

PYBIND11_MODULE(hnsw_cpp, m) {
    m.doc() = "HNSW module";
    py::class_<HNSWindex>(m, "HNSWindex")
        .def(py::init<int, int, int, int, int>(), py::arg("dim"), py::arg("m"),
             py::arg("efConstruction"), py::arg("efSearch"),
             py::arg("flag") = 0)
        .def(
            "Insert",
            [](HNSWindex& self,
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   data) {
                auto buf = data.request();
                size_t n = buf.shape[0];
                float* ptr = static_cast<float*>(buf.ptr);
                {
                    py::gil_scoped_release release;
                    self.Insert(ptr, n);
                }
            },
            py::arg("nq"))
        .def(
            "Search",
            [](HNSWindex& self, at::Tensor data, int K, int ef = -1) {
                auto buf = data.contiguous();
                auto ptr = buf.data_ptr<float>();
                int64_t n = buf.size(0);
                auto optionFloat =
                    at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
                auto optionInt =
                    at::TensorOptions().dtype(at::kInt).device(at::kCPU);
                // -1 填充：索引为空或候选不足 K 时，剩余位置不会是未初始化的垃圾值
                auto D = torch::full({n, K}, -1.0f, optionFloat);
                auto I = torch::full({n, K}, -1, optionInt);
                auto Dptr = D.data_ptr<float>();
                auto Iptr = I.data_ptr<int>();
                {
                    py::gil_scoped_release release;
                    self.KNNSearch(ptr, static_cast<int>(n), K, Dptr, Iptr,
                                   ef);
                }
                return std::make_tuple(D, I);
            },
            py::arg("q"), py::arg("k"), py::arg("efSearch") = -1);

    // 兼容旧名字：hnsw_cpp.HNSW 和 hnsw_cpp.HNSWindex 是同一个类型
    m.attr("HNSW") = m.attr("HNSWindex");
}
