### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 24f463ee-76e6-11ef-0b34-796df7c80881
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 0fd3fb14-34f4-419b-bfe4-9eff238cc439
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO

# ╔═╡ 8376c0c9-7030-4425-83ad-be7c99609b7d
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 62790b6a-c264-44f0-aeca-8116bd9a8471
md"""
### Loading Julia Packages
"""

# ╔═╡ 46bc740c-8109-4cb0-931b-099b29154426
@bind City Select(["abudhabi", "mumbai"])

# ╔═╡ 40be7a21-175b-4032-af72-3b2258117402
md"""
### Storing the directory path for MSI files in Img1 and Img2 variables
"""

# ╔═╡ ab3e1205-ed61-408f-968f-9b91a572005e
Img1_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect")

# ╔═╡ 692b666b-b45b-4063-a7e4-4f5d494443fe
Img2_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_2_rect")

# ╔═╡ 41d6de1e-0fdb-4658-a0ee-aa87b0366c6e
md"""
### Storing all 13 *.tif files in I1 and I2 variables
"""

# ╔═╡ 67de718c-65f5-4b87-b31e-6915316778e5
I1 = glob("*.tif", Img1_Path)

# ╔═╡ 259b1fc5-a3ac-4bbb-bb93-73b48eebd780
I2 = glob("*.tif", Img2_Path)

# ╔═╡ 59260065-50e7-4f4d-9460-ae77c63ac34b
GT1_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"pair",
	"img1.png")

# ╔═╡ 9fa99b74-510e-415b-9db8-5954a359fc02
GT2_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"pair",
	"img2.png")

# ╔═╡ 06d2f65b-73bc-4114-8c21-70d1e36e4aaf
md"""
#### Al 13 matrices are stored in the Img1 and Img2 variables
"""

# ╔═╡ f893213d-ef21-4b33-84de-8e3a51cd71f4
Img1 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I1]

# ╔═╡ 4e390cf0-016e-4b83-b3ef-6c66ee325c69
Img2 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I2]

# ╔═╡ 551bccd4-54e0-4f7e-af41-a6063889e464
tens1 = cat(Img1..., dims=3);

# ╔═╡ a2d1e360-b7b0-4301-9375-11fba3ecf911
@bind Band PlutoUI.Slider(1:13, show_value=true)

# ╔═╡ cde91b75-74ff-45d9-b3c0-46873e8a8be7
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax, tens1[:, :, Band])
	fig
end

# ╔═╡ 82770fa3-323b-4d7c-b8e8-2aad72c89858
tens2 = cat(Img2..., dims=3);

# ╔═╡ bb5b4703-eb92-47ae-9d23-c1db7c096e6f
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax, tens2[:, :, Band])
	fig
end

# ╔═╡ a82836bc-fba3-4aad-b56f-14ec2f866417
GT1 = load(GT1_Path)

# ╔═╡ 80474dfe-0bf6-408d-8431-3d55612c6411
GT2 = load(GT2_Path)

# ╔═╡ e1e2606e-1c2c-4ea9-8227-b3f93a2412af
bw1_gt = Gray.(GT1)

# ╔═╡ 5c9157e0-f1d8-4e39-b4aa-2e482366e053
bw2_gt = Gray.(GT2)

# ╔═╡ 56ba9e6b-8dba-4f83-8ca2-20aa26b9571e
diff_img = abs.(bw2_gt .- bw1_gt)

# ╔═╡ 9df61fab-d8ab-4712-9f0f-bb428e119d3f
binary_change_map = diff_img .> 0.8

# ╔═╡ d193adf7-eead-4572-972b-d90bfebc482c


# ╔═╡ 91ae7529-fcea-45c7-89f3-16e8795f0013
with_theme() do
	fig = Figure(; size=(600, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, binary_change_map)
	fig
end

# ╔═╡ Cell order:
# ╠═8376c0c9-7030-4425-83ad-be7c99609b7d
# ╟─fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
# ╠═24f463ee-76e6-11ef-0b34-796df7c80881
# ╟─62790b6a-c264-44f0-aeca-8116bd9a8471
# ╠═0fd3fb14-34f4-419b-bfe4-9eff238cc439
# ╠═46bc740c-8109-4cb0-931b-099b29154426
# ╟─40be7a21-175b-4032-af72-3b2258117402
# ╠═ab3e1205-ed61-408f-968f-9b91a572005e
# ╠═692b666b-b45b-4063-a7e4-4f5d494443fe
# ╟─41d6de1e-0fdb-4658-a0ee-aa87b0366c6e
# ╠═67de718c-65f5-4b87-b31e-6915316778e5
# ╠═259b1fc5-a3ac-4bbb-bb93-73b48eebd780
# ╠═59260065-50e7-4f4d-9460-ae77c63ac34b
# ╠═9fa99b74-510e-415b-9db8-5954a359fc02
# ╠═06d2f65b-73bc-4114-8c21-70d1e36e4aaf
# ╠═f893213d-ef21-4b33-84de-8e3a51cd71f4
# ╠═4e390cf0-016e-4b83-b3ef-6c66ee325c69
# ╠═551bccd4-54e0-4f7e-af41-a6063889e464
# ╠═a2d1e360-b7b0-4301-9375-11fba3ecf911
# ╠═cde91b75-74ff-45d9-b3c0-46873e8a8be7
# ╠═82770fa3-323b-4d7c-b8e8-2aad72c89858
# ╠═bb5b4703-eb92-47ae-9d23-c1db7c096e6f
# ╠═a82836bc-fba3-4aad-b56f-14ec2f866417
# ╠═80474dfe-0bf6-408d-8431-3d55612c6411
# ╠═e1e2606e-1c2c-4ea9-8227-b3f93a2412af
# ╠═5c9157e0-f1d8-4e39-b4aa-2e482366e053
# ╠═56ba9e6b-8dba-4f83-8ca2-20aa26b9571e
# ╠═9df61fab-d8ab-4712-9f0f-bb428e119d3f
# ╠═d193adf7-eead-4572-972b-d90bfebc482c
# ╠═91ae7529-fcea-45c7-89f3-16e8795f0013
