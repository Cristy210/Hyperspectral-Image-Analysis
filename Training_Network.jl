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
cities = ["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"]

# ╔═╡ d43d3362-de4c-4a3d-aca6-483e470d68a6
data_path_T1 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect") for City in cities]

# ╔═╡ 88a13234-6e0a-4594-966e-36f1e4d8f2a9
data_path_T2 = [joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_2_rect") for City in cities]

# ╔═╡ e62184d8-4e01-4374-a60e-d035eaefd3ae
md"""
### Storing all .tif files from the path
"""

# ╔═╡ 8fcfb326-41b2-49b9-9968-f122434a7874
Tif1_Path = [glob("*.tif", Img1_Path) for Img1_Path in data_path_T1]

# ╔═╡ f833aa65-ed88-400f-bb9a-4a6531885c41
Tif2_Path = [glob("*.tif", Img2_Path) for Img2_Path in data_path_T2]

# ╔═╡ a7c27379-0cd3-4667-8c79-6d519b34084a
T1 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I1] for I1 in Tif1_Path]

# ╔═╡ 13728c51-6d86-4494-b984-6ae06c731b71
T2 = [[ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I2] for I2 in Tif2_Path]

# ╔═╡ 95ae8f14-363c-493c-a7ee-3e5c0a38b883
Temp1_tens = [cat(V1..., dims=3) for V1 in T1]

# ╔═╡ 8e424cd0-c424-485d-959c-4c7cb4c16a06
Temp2_tens = [cat(V2..., dims=3) for V2 in T2]

# ╔═╡ Cell order:
# ╠═e9b02eb0-7c82-11ef-36cd-b7cf22cfd23c
# ╟─1a119f4c-880c-46e5-a528-a673adbcc11e
# ╠═67862dcc-aa79-46af-998b-7c023a7a3164
# ╠═6173b170-62fe-4a15-8520-20fe7ba06c5d
# ╟─52f6d430-ed43-48d8-9670-599fc125ec95
# ╠═f5c64e34-664c-4a86-94bc-f1b86d250f0f
# ╠═d43d3362-de4c-4a3d-aca6-483e470d68a6
# ╠═88a13234-6e0a-4594-966e-36f1e4d8f2a9
# ╟─e62184d8-4e01-4374-a60e-d035eaefd3ae
# ╠═8fcfb326-41b2-49b9-9968-f122434a7874
# ╠═f833aa65-ed88-400f-bb9a-4a6531885c41
# ╠═a7c27379-0cd3-4667-8c79-6d519b34084a
# ╠═13728c51-6d86-4494-b984-6ae06c731b71
# ╠═95ae8f14-363c-493c-a7ee-3e5c0a38b883
# ╠═8e424cd0-c424-485d-959c-4c7cb4c16a06
