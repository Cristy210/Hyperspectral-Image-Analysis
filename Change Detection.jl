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
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO

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
### List of Cities
"""

# ╔═╡ f2c55eb8-4bd7-486c-afb7-5fcb656674a7
@bind City Select(["abudhabi", "mumbai", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "cupertino", "hongkong", "nantes", "paris", "pisa", "rennes", "saclay_e"])

# ╔═╡ 527b195e-ca5a-4d93-99fa-1e18577c17da
label_path = joinpath(@__DIR__, 
	"OSCD Train",
	City,
	"cm", 
	"$City-cm.tif")

# ╔═╡ 60ca98e3-dda2-44ae-8fd7-f5326f9444ee
ds = ArchGDAL.read(label_path)

# ╔═╡ e5e40029-3faa-40b1-bb73-113fe989f47b
band = ArchGDAL.getband(ds, 1)

# ╔═╡ 81278fc3-b5f1-41d6-9c42-0d6380b57827
Img = ArchGDAL.read(band)

# ╔═╡ c3a29a46-52a3-48d0-97a7-24d30d7fed87
md"""
### Binary Change Map for Training Dataset
"""

# ╔═╡ 5f404cec-7b3c-49df-8dd8-ffce534506cd
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, Img)
	fig
end

# ╔═╡ aa7f1f63-cd04-4db6-9b87-6af718a81489


# ╔═╡ Cell order:
# ╟─3d1ecbd3-e3cd-43c2-83e3-48bc7468ec8c
# ╟─10cdbf70-7ba3-11ef-2872-77bdaddec0c7
# ╠═07455ec4-b6f7-423f-949a-3cc69d342455
# ╟─4217344b-de62-40c3-a759-1b55d661c2f3
# ╠═17144ed9-b04e-4034-ad4b-afbdbf16270f
# ╟─051472d7-b348-48d8-83d2-2ea524808d05
# ╠═f2c55eb8-4bd7-486c-afb7-5fcb656674a7
# ╠═527b195e-ca5a-4d93-99fa-1e18577c17da
# ╠═60ca98e3-dda2-44ae-8fd7-f5326f9444ee
# ╠═e5e40029-3faa-40b1-bb73-113fe989f47b
# ╠═81278fc3-b5f1-41d6-9c42-0d6380b57827
# ╟─c3a29a46-52a3-48d0-97a7-24d30d7fed87
# ╠═5f404cec-7b3c-49df-8dd8-ffce534506cd
# ╠═aa7f1f63-cd04-4db6-9b87-6af718a81489
