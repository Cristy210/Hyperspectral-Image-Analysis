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

# ╔═╡ 07455ec4-b6f7-423f-949a-3cc69d342455
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 17144ed9-b04e-4034-ad4b-afbdbf16270f
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO, Interpolations, PythonCall, PaddedViews

# ╔═╡ 3d1ecbd3-e3cd-43c2-83e3-48bc7468ec8c
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 10cdbf70-7ba3-11ef-2872-77bdaddec0c7
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ 4217344b-de62-40c3-a759-1b55d661c2f3
md"""
### Loading Julia Packages
"""

# ╔═╡ 051472d7-b348-48d8-83d2-2ea524808d05
md"""
## List of Cities - Training Data
"""

# ╔═╡ f2c55eb8-4bd7-486c-afb7-5fcb656674a7
@bind City Select(["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"])

# ╔═╡ 527b195e-ca5a-4d93-99fa-1e18577c17da
label_path = joinpath(@__DIR__, 
	"OSCD Train",
	City,
	"cm", 
	"$City-cm.tif")

# ╔═╡ 73810fdb-a0fc-4d3e-a628-9fc5e691485d
md"""
### Storing the directory path for Bi-Temporal MSI files in Img1 and Img2 variables
"""

# ╔═╡ 3008a09a-2ce9-4155-9e53-ddc5b40b0499
Img1_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect")

# ╔═╡ b9b93793-201d-4528-aa46-44a341778931
Img2_Path = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_2_rect")

# ╔═╡ c950bb4e-c0c5-4cdd-9578-fad30913d4ed


# ╔═╡ 8093dbed-055b-41a9-ab16-0de9488167fa
md"""
### Storing all 13 *.tif files in I1 and I2 variables using Glob Package
"""

# ╔═╡ e6acbe17-d2c9-448b-879b-33a377a7314a
I1 = glob("*.tif", Img1_Path)

# ╔═╡ 7e9144f0-2c5f-43cf-8ee6-1f46765a7732
I2 = glob("*.tif", Img2_Path)

# ╔═╡ 94c0bf37-d327-4468-8e9b-80e8f58efff6
md"""
#### Al 13 Images are stored in the T1 and T2 variables
"""

# ╔═╡ b7900fe1-21a8-4334-89f7-4618432d4d8d
T1 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I1]

# ╔═╡ 5343ec45-b91d-4953-b000-1e551c204322
T2 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in I2]

# ╔═╡ f0f35ce2-e1fb-4c13-9e83-9821b49dc809
tens1 = cat(T1..., dims=3); tens2 = cat(T2..., dims=3)

# ╔═╡ d64d0143-31b8-4a73-acac-1dee8ab3b29b
@bind Band PlutoUI.Slider(1:13, show_value=true)

# ╔═╡ a14b6bb4-00c4-4ba6-b50c-a759ae61ec30
# Makie.available_gradients()

# ╔═╡ 72f6c3eb-1125-4c38-8e2e-7fb1ba621e74
with_theme() do
	fig = Figure(; size=(800, 600))
	ax1 = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax1, tens1[:, :, Band])
	ax2 = Axis(fig[1, 2], aspect = DataAspect(), yreversed=true)
	image!(ax2, tens2[:, :, Band])
	fig
end

# ╔═╡ 60ca98e3-dda2-44ae-8fd7-f5326f9444ee
ds = ArchGDAL.read(label_path)

# ╔═╡ e5e40029-3faa-40b1-bb73-113fe989f47b
band = ArchGDAL.getband(ds, 1)

# ╔═╡ 81278fc3-b5f1-41d6-9c42-0d6380b57827
BM = ArchGDAL.read(band)

# ╔═╡ c3a29a46-52a3-48d0-97a7-24d30d7fed87
md"""
### Binary Change Map for Training Dataset
"""

# ╔═╡ 5f404cec-7b3c-49df-8dd8-ffce534506cd
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, BM)
	fig
end

# ╔═╡ aa7f1f63-cd04-4db6-9b87-6af718a81489
x_size, y_size, bands = size(tens1)

# ╔═╡ 1fb4acdf-a121-4319-ae51-818b353ef8e8
x_pad = 15 - (x_size % 15)

# ╔═╡ a258a2a0-cb87-4280-b8ba-39dafdea01ee
y_pad = 15 - (y_size % 15)

# ╔═╡ e7cebeb8-fd22-40e0-ab68-a04f202b791a
PD_Array = PaddedView(0, tens1, ((x_size + x_pad), (y_size + y_pad), bands))

# ╔═╡ baf391d3-50f1-4910-a63b-bec093683b0e
with_theme() do
	fig = Figure(; size=(800, 600))
	ax1 = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax1, PD_Array[:, :, Band])
	fig
end

# ╔═╡ Cell order:
# ╠═3d1ecbd3-e3cd-43c2-83e3-48bc7468ec8c
# ╠═10cdbf70-7ba3-11ef-2872-77bdaddec0c7
# ╠═07455ec4-b6f7-423f-949a-3cc69d342455
# ╟─4217344b-de62-40c3-a759-1b55d661c2f3
# ╠═17144ed9-b04e-4034-ad4b-afbdbf16270f
# ╟─051472d7-b348-48d8-83d2-2ea524808d05
# ╠═f2c55eb8-4bd7-486c-afb7-5fcb656674a7
# ╠═527b195e-ca5a-4d93-99fa-1e18577c17da
# ╟─73810fdb-a0fc-4d3e-a628-9fc5e691485d
# ╠═3008a09a-2ce9-4155-9e53-ddc5b40b0499
# ╠═b9b93793-201d-4528-aa46-44a341778931
# ╠═c950bb4e-c0c5-4cdd-9578-fad30913d4ed
# ╟─8093dbed-055b-41a9-ab16-0de9488167fa
# ╠═e6acbe17-d2c9-448b-879b-33a377a7314a
# ╠═7e9144f0-2c5f-43cf-8ee6-1f46765a7732
# ╟─94c0bf37-d327-4468-8e9b-80e8f58efff6
# ╠═b7900fe1-21a8-4334-89f7-4618432d4d8d
# ╠═5343ec45-b91d-4953-b000-1e551c204322
# ╠═f0f35ce2-e1fb-4c13-9e83-9821b49dc809
# ╠═d64d0143-31b8-4a73-acac-1dee8ab3b29b
# ╠═a14b6bb4-00c4-4ba6-b50c-a759ae61ec30
# ╠═72f6c3eb-1125-4c38-8e2e-7fb1ba621e74
# ╠═60ca98e3-dda2-44ae-8fd7-f5326f9444ee
# ╠═e5e40029-3faa-40b1-bb73-113fe989f47b
# ╠═81278fc3-b5f1-41d6-9c42-0d6380b57827
# ╟─c3a29a46-52a3-48d0-97a7-24d30d7fed87
# ╠═5f404cec-7b3c-49df-8dd8-ffce534506cd
# ╠═aa7f1f63-cd04-4db6-9b87-6af718a81489
# ╠═1fb4acdf-a121-4319-ae51-818b353ef8e8
# ╠═a258a2a0-cb87-4280-b8ba-39dafdea01ee
# ╠═e7cebeb8-fd22-40e0-ab68-a04f202b791a
# ╠═baf391d3-50f1-4910-a63b-bec093683b0e
