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
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging

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

# ╔═╡ 18e316a8-9ec6-4343-8cf4-7a04eccd0df8
@bind item Select(["coffee", "rice", "sugar_salt_flour", "sugar_salt_flour_contamination", "yatsuhashi"])

# ╔═╡ 4d453d4e-c7b0-4af0-9df4-16957539c621
n_cluster_map = Dict(
	"coffee" => 7,
	"rice" => 4,
	"sugar_salt_flour" => 3,
	"sugar_salt_flour_contamination" => 4,
	"yatsuhashi" => 3
)

# ╔═╡ 2f419b6e-ae3f-42b3-b854-8a290e3239cd
n_clusters = n_cluster_map[item]

# ╔═╡ 40be7a21-175b-4032-af72-3b2258117402
md"""
### Storing the directory path for HSI files
"""

# ╔═╡ 51d5b05d-683c-4d96-bc16-a6ef972f0d04
data_path = joinpath(@__DIR__, "NIR_HSI_Coffee_DS", "data", item)

# ╔═╡ 74af4aed-0e24-4045-90ca-aa14a0284a44
files = glob("*.png", data_path)

# ╔═╡ b856c94b-1bac-49be-aeb2-25da1d8e75e9
wavelengths = [parse(Int, match(r"(\d+)nm", file).captures[1]) for file in files]

# ╔═╡ 4dab33d9-3b14-4615-88a5-11fa0239ba65
Array = [Float64.(load(files[i])) for i in 1:length(files)]

# ╔═╡ 4e93dad5-c25b-4f34-bfe7-e244be1637e2
function hsi2rgb(hsicube, wavelengths)
	# Identify wavelengths for RGB
	rgb_idx = [
		findmin(w -> abs(w-615), wavelengths)[2],
		findmin(w -> abs(w-520), wavelengths)[2],
		findmin(w -> abs(w-450), wavelengths)[2],
	]

	# Extract bands and clamp
	rgbcube = clamp!(hsicube[:, :, rgb_idx], 0, 1)

	# Form matrix of RGB values
	return Makie.RGB.(
		view(rgbcube, :, :, 1),
		view(rgbcube, :, :, 2),
		view(rgbcube, :, :, 3),
	)
end

# ╔═╡ 8207d5c3-6945-40bc-bb5e-ba7db29a751d
begin
hsi2rgba(alpha, hsicube, wavelengths) = Makie.RGBA.(hsi2rgb(hsicube, wavelengths), alpha)
hsi2rgba(alpha_func::Function, hsicube, wavelengths) = hsi2rgba(
	map(alpha_func, eachslice(hsicube; dims=(1,2))),
	hsicube, wavelengths,
)
end

# ╔═╡ f1b52435-459b-442a-a2ce-0e98d3dd550a
THEME = Theme(; backgroundcolor=(:black, 0), textcolor=:white, Legend=(; backgroundcolor=(:black, 0), framevisible=false))
# THEME = Theme(;)

# ╔═╡ 3018818b-a409-46d4-9cd1-92f6113006dc
CACHEDIR = joinpath(@__DIR__, "cache_files")

# ╔═╡ b67c80d7-be13-4e79-a14b-9aa6fea7eb78
data = cat(Array..., dims=3)

# ╔═╡ 62d7769b-dcdc-466b-84a7-df500ffe4b16
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ b4d1ea02-c7ad-4269-acc2-d34ea75acd26
data_refined = data[10:256,30:180, :]

# ╔═╡ def9cd14-3a06-45f6-b85b-2a4aa04c8fda
data_refined[50, 50, :]

# ╔═╡ 18262d61-d4d2-457e-a79a-77c3a365042c
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 88fa84cd-b785-4bf7-841e-3c623c381c86
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data_refined[:, :, band]))
	fig
end

# ╔═╡ c0d4e03b-ffe4-4456-b35b-298d5ae31e63
mask = map(s -> norm(s) < 3.250, eachslice(data_refined; dims=(1,2)))

# ╔═╡ c4cecb4e-f5b1-493f-8170-f387c38dd8fd
with_theme(THEME) do
	fig = Figure(; size=(350, 500))
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(hsi2rgba(mask,data_refined,wavelengths)))
	fig
end

# ╔═╡ 8a82b7ae-dccd-4375-b6c4-11bd080eff4a
permutedims(data_refined[mask, :])

# ╔═╡ 5fde82bd-73bd-4528-8116-853719d74fd0
md"""
#### Compute Affinity Matrix
"""

# ╔═╡ Cell order:
# ╠═8376c0c9-7030-4425-83ad-be7c99609b7d
# ╠═fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
# ╠═24f463ee-76e6-11ef-0b34-796df7c80881
# ╟─62790b6a-c264-44f0-aeca-8116bd9a8471
# ╠═0fd3fb14-34f4-419b-bfe4-9eff238cc439
# ╠═18e316a8-9ec6-4343-8cf4-7a04eccd0df8
# ╠═4d453d4e-c7b0-4af0-9df4-16957539c621
# ╠═2f419b6e-ae3f-42b3-b854-8a290e3239cd
# ╟─40be7a21-175b-4032-af72-3b2258117402
# ╠═51d5b05d-683c-4d96-bc16-a6ef972f0d04
# ╠═74af4aed-0e24-4045-90ca-aa14a0284a44
# ╠═b856c94b-1bac-49be-aeb2-25da1d8e75e9
# ╠═4dab33d9-3b14-4615-88a5-11fa0239ba65
# ╠═4e93dad5-c25b-4f34-bfe7-e244be1637e2
# ╠═8207d5c3-6945-40bc-bb5e-ba7db29a751d
# ╠═f1b52435-459b-442a-a2ce-0e98d3dd550a
# ╠═3018818b-a409-46d4-9cd1-92f6113006dc
# ╠═b67c80d7-be13-4e79-a14b-9aa6fea7eb78
# ╠═62d7769b-dcdc-466b-84a7-df500ffe4b16
# ╠═b4d1ea02-c7ad-4269-acc2-d34ea75acd26
# ╠═def9cd14-3a06-45f6-b85b-2a4aa04c8fda
# ╠═18262d61-d4d2-457e-a79a-77c3a365042c
# ╠═88fa84cd-b785-4bf7-841e-3c623c381c86
# ╠═c0d4e03b-ffe4-4456-b35b-298d5ae31e63
# ╠═c4cecb4e-f5b1-493f-8170-f387c38dd8fd
# ╠═8a82b7ae-dccd-4375-b6c4-11bd080eff4a
# ╠═5fde82bd-73bd-4528-8116-853719d74fd0
