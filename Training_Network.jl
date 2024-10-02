### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 67862dcc-aa79-46af-998b-7c023a7a3164
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 6173b170-62fe-4a15-8520-20fe7ba06c5d
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO, Interpolations, PythonCall, PaddedViews

# ╔═╡ e9b02eb0-7c82-11ef-36cd-b7cf22cfd23c
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 1a119f4c-880c-46e5-a528-a673adbcc11e
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ 52f6d430-ed43-48d8-9670-599fc125ec95
md"""
## List of Cities - Training Data
"""

# ╔═╡ f5c64e34-664c-4a86-94bc-f1b86d250f0f
train_cities = ["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"]

# ╔═╡ d43d3362-de4c-4a3d-aca6-483e470d68a6
data_path_T1 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect") for City in train_cities]

# ╔═╡ 88a13234-6e0a-4594-966e-36f1e4d8f2a9
data_path_T2 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_2_rect") for City in train_cities]

# ╔═╡ d3121ede-d28a-4e45-824f-c67a67f57d98
train_label_path = [joinpath(@__DIR__, 
	"OSCD Train",
	City,
	"cm", 
	"$City-cm.tif") for City in train_cities]

# ╔═╡ cf5cf0ef-4c9b-44b1-a760-cc7c4e953968


# ╔═╡ 4039c03e-2f49-4bdd-b486-a66591bc3724
md"""
### List of Cities - Testing Data
"""

# ╔═╡ 0103b620-d8f7-436d-9e80-6a8b6cc1830e
test_cities = ["brasilia", "chongqing", "dubai", "lasvegas", "milano", "montpellier", "norcia", "rio", "saclay_w", "valencia"]

# ╔═╡ 51a836e3-c493-4d8d-8d47-60c6b1ebb975
test_data_path_T1 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect") for City in test_cities]

# ╔═╡ c60ee822-f116-48f1-b9e9-d876484b34da
test_data_path_T2 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_2_rect") for City in test_cities]

# ╔═╡ 1e7f3d9f-22b3-4f73-a614-530b43679783
test_label_path = [joinpath(@__DIR__, 
	"OSCD Test",
	City,
	"cm", 
	"$City-cm.tif") for City in test_cities]

# ╔═╡ 552c7eb1-de11-44d0-ba89-1d88b25e2015


# ╔═╡ e62184d8-4e01-4374-a60e-d035eaefd3ae
md"""
### Storing all .tif files from the path - Training Data
"""

# ╔═╡ 8fcfb326-41b2-49b9-9968-f122434a7874
Tif1_Path = [glob("*.tif", Img1_Path) for Img1_Path in data_path_T1]

# ╔═╡ f833aa65-ed88-400f-bb9a-4a6531885c41
Tif2_Path = [glob("*.tif", Img2_Path) for Img2_Path in data_path_T2]

# ╔═╡ a7c27379-0cd3-4667-8c79-6d519b34084a
T1 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I1] for I1 in Tif1_Path]

# ╔═╡ 50c3dae3-1893-46a0-9bd6-19f1de54a5f0
with_theme() do
	fig = Figure(; size=(600, 400))
	ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax, T1[1][2])
	fig
end

# ╔═╡ 13728c51-6d86-4494-b984-6ae06c731b71
T2 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I2] for I2 in Tif2_Path]

# ╔═╡ 95ae8f14-363c-493c-a7ee-3e5c0a38b883
Temp1_tens = [cat(V1..., dims=3) for V1 in T1]

# ╔═╡ 8e424cd0-c424-485d-959c-4c7cb4c16a06
Temp2_tens = [cat(V2..., dims=3) for V2 in T2]

# ╔═╡ e2434a72-0f81-472a-8474-af91fb294950
md"""
#### Trainig Data - Ground Truth
"""

# ╔═╡ 372a329c-182a-4eca-99ef-f58a90c37625
train_gt = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end] for tif_file in train_label_path]

# ╔═╡ 9320f572-7dca-4364-b7e0-9a6a10e81a97
md"""
### Storing all .tif files from the path - Testing Data
"""

# ╔═╡ d2f49d62-78d1-4e31-856c-11fc8b442ab8
test_Tif1_Path = [glob("*.tif", Img1_Path) for Img1_Path in test_data_path_T1]

# ╔═╡ f29beb16-2894-4e35-9839-8f89da9ab240
test_Tif2_Path = [glob("*.tif", Img2_Path) for Img2_Path in test_data_path_T2]

# ╔═╡ 46e5ee3c-8669-48e2-8bca-c16791c8cffd
test_T1 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I1] for I1 in test_Tif1_Path]

# ╔═╡ 83509446-b4ce-45dc-a70b-4872afe59d5d
test_T2 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I2] for I2 in test_Tif2_Path]

# ╔═╡ 7f37669c-2ad0-42d0-828a-28c51310b03e
test_Temp1_tens = [cat(V1..., dims=3) for V1 in test_T1]

# ╔═╡ 12b45f01-42ea-47e1-9521-efdbfde3b7fd
test_Temp2_tens = [cat(V2..., dims=3) for V2 in test_T2]

# ╔═╡ 9d913122-4492-425b-832e-c4a4b1ebcc4b
md"""
#### Testing Data - Ground Truth
"""

# ╔═╡ 1043e275-83d3-4300-8441-f30cb055a564
test_gt = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end] for tif_file in test_label_path]

# ╔═╡ be3475a5-c89f-4dd6-b8e3-cac7ca0237c1


# ╔═╡ Cell order:
# ╟─e9b02eb0-7c82-11ef-36cd-b7cf22cfd23c
# ╟─1a119f4c-880c-46e5-a528-a673adbcc11e
# ╠═67862dcc-aa79-46af-998b-7c023a7a3164
# ╠═6173b170-62fe-4a15-8520-20fe7ba06c5d
# ╟─52f6d430-ed43-48d8-9670-599fc125ec95
# ╠═f5c64e34-664c-4a86-94bc-f1b86d250f0f
# ╠═d43d3362-de4c-4a3d-aca6-483e470d68a6
# ╠═88a13234-6e0a-4594-966e-36f1e4d8f2a9
# ╠═d3121ede-d28a-4e45-824f-c67a67f57d98
# ╠═cf5cf0ef-4c9b-44b1-a760-cc7c4e953968
# ╟─4039c03e-2f49-4bdd-b486-a66591bc3724
# ╠═0103b620-d8f7-436d-9e80-6a8b6cc1830e
# ╠═51a836e3-c493-4d8d-8d47-60c6b1ebb975
# ╠═c60ee822-f116-48f1-b9e9-d876484b34da
# ╠═1e7f3d9f-22b3-4f73-a614-530b43679783
# ╠═552c7eb1-de11-44d0-ba89-1d88b25e2015
# ╟─e62184d8-4e01-4374-a60e-d035eaefd3ae
# ╠═8fcfb326-41b2-49b9-9968-f122434a7874
# ╠═f833aa65-ed88-400f-bb9a-4a6531885c41
# ╠═a7c27379-0cd3-4667-8c79-6d519b34084a
# ╠═50c3dae3-1893-46a0-9bd6-19f1de54a5f0
# ╠═13728c51-6d86-4494-b984-6ae06c731b71
# ╠═95ae8f14-363c-493c-a7ee-3e5c0a38b883
# ╠═8e424cd0-c424-485d-959c-4c7cb4c16a06
# ╟─e2434a72-0f81-472a-8474-af91fb294950
# ╠═372a329c-182a-4eca-99ef-f58a90c37625
# ╟─9320f572-7dca-4364-b7e0-9a6a10e81a97
# ╠═d2f49d62-78d1-4e31-856c-11fc8b442ab8
# ╠═f29beb16-2894-4e35-9839-8f89da9ab240
# ╠═46e5ee3c-8669-48e2-8bca-c16791c8cffd
# ╠═83509446-b4ce-45dc-a70b-4872afe59d5d
# ╠═7f37669c-2ad0-42d0-828a-28c51310b03e
# ╠═12b45f01-42ea-47e1-9521-efdbfde3b7fd
# ╟─9d913122-4492-425b-832e-c4a4b1ebcc4b
# ╠═1043e275-83d3-4300-8441-f30cb055a564
# ╠═be3475a5-c89f-4dd6-b8e3-cac7ca0237c1
