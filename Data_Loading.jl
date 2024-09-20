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
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL

# ╔═╡ fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 7e4e2208-d1fd-43d2-aaeb-0a93aa375612


# ╔═╡ 46bc740c-8109-4cb0-931b-099b29154426
@bind City Select(["abudhabi", "mumbai"])

# ╔═╡ ab3e1205-ed61-408f-968f-9b91a572005e
Img1 = joinpath(@__DIR__, 
	"Onera Satellite Change Detection dataset - Images",
	City,
	"imgs_1_rect")

# ╔═╡ 7bd27196-19ae-4dcc-a9fb-f26ecd34c6da
Img1_tif_files = readdir(Img1; join=true) |> filter(f -> endswith(f, ".tif"))

# ╔═╡ f255aac9-b157-4a49-ae14-b757754a1d5d
Img1_tif_files[1]

# ╔═╡ a644b8e8-bc08-4003-a85b-b5eba6f5a7bc
ArchGDAL.read(Img1_tif_files[1])

# ╔═╡ ced11ff2-534a-4a37-a44a-1640685924e7
Images1 = []

# ╔═╡ 4529796c-f67f-415d-b08f-32557994e3f1
for file in Img1_tif_files
    ArchGDAL.read(file) do dataset
        band = ArchGDAL.getband(dataset, 1)
        data = ArchGDAL.read(band)
        push!(Images1, data)
    end
end

# ╔═╡ 8aa888bb-691b-4420-8bcc-8a0580160b0c
Images1

# ╔═╡ 551bccd4-54e0-4f7e-af41-a6063889e464
tens = cat(Images1..., dims=3)

# ╔═╡ cde91b75-74ff-45d9-b3c0-46873e8a8be7
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax, tens[:, :, 3])
	fig
end

# ╔═╡ ba8896e2-24eb-463c-9f0c-1ee3d52dfc9d


# ╔═╡ Cell order:
# ╟─fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
# ╠═24f463ee-76e6-11ef-0b34-796df7c80881
# ╠═0fd3fb14-34f4-419b-bfe4-9eff238cc439
# ╠═7e4e2208-d1fd-43d2-aaeb-0a93aa375612
# ╠═46bc740c-8109-4cb0-931b-099b29154426
# ╠═ab3e1205-ed61-408f-968f-9b91a572005e
# ╠═7bd27196-19ae-4dcc-a9fb-f26ecd34c6da
# ╠═f255aac9-b157-4a49-ae14-b757754a1d5d
# ╠═a644b8e8-bc08-4003-a85b-b5eba6f5a7bc
# ╠═ced11ff2-534a-4a37-a44a-1640685924e7
# ╠═4529796c-f67f-415d-b08f-32557994e3f1
# ╠═8aa888bb-691b-4420-8bcc-8a0580160b0c
# ╠═551bccd4-54e0-4f7e-af41-a6063889e464
# ╠═cde91b75-74ff-45d9-b3c0-46873e8a8be7
# ╠═ba8896e2-24eb-463c-9f0c-1ee3d52dfc9d
