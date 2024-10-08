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
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO, ArnoldiMethod, CacheVariables

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

# ╔═╡ 40be7a21-175b-4032-af72-3b2258117402
md"""
### Storing the directory path for HSI files
"""

# ╔═╡ 51d5b05d-683c-4d96-bc16-a6ef972f0d04
data_path = joinpath(@__DIR__, "NIR_HSI_Coffee_DS", "data", item)

# ╔═╡ 74af4aed-0e24-4045-90ca-aa14a0284a44
files = glob("*.png", data_path)

# ╔═╡ 4dab33d9-3b14-4615-88a5-11fa0239ba65
Array = [Float64.(load(files[i])) for i in 1:length(files)]

# ╔═╡ b67c80d7-be13-4e79-a14b-9aa6fea7eb78
data_tens = cat(Array..., dims=3)

# ╔═╡ a6b8a27b-614c-42ba-941e-792f586b2881
trimmed = data_tens[20:end, 30:end, :]

# ╔═╡ 18262d61-d4d2-457e-a79a-77c3a365042c
@bind band PlutoUI.Slider(1:size(data_tens, 3), show_value=true)

# ╔═╡ 88fa84cd-b785-4bf7-841e-3c623c381c86
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data_tens[:, :, band]))
	fig
end

# ╔═╡ 2b027eda-4ec5-4315-bc7e-5a7774abac9b
data_mat = permutedims(reshape(data_tens, :, size(data_tens, 3)))

# ╔═╡ 6794a588-7c72-47dd-8395-164545b6132a
col_norms = [norm(data_mat[:, i]) for i in 1:size(data_mat, 2)]

# ╔═╡ 3835fef4-ab2a-44bf-a3b5-0bce1a70c605
Norm_vec = [data_mat[:, i] ./ col_norms[i] for i in 1:size(data_mat, 2)]

# ╔═╡ 0050899c-2616-4f7f-aac7-c0a2eefb6ca2
Norm_mat = hcat(Norm_vec...)

# ╔═╡ 36c193e0-7531-4120-91e0-3e0ecdcb65bc
# A = transpose(Norm_mat) * Norm_mat

# ╔═╡ 0d12ebec-75c3-4f15-8098-9867bb6a3626
# S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

# ╔═╡ b81de22d-469a-41e4-acf4-538d395bc653
# diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ 4f58a7ac-ae6e-4e43-9d59-8a5d1eab6554
# D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ 9982c4a7-bd44-4f04-a241-2eeeefaaa0fb
# L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ a59ce994-3c51-4613-ac10-7fda447ef3af


# ╔═╡ Cell order:
# ╠═8376c0c9-7030-4425-83ad-be7c99609b7d
# ╟─fa6f6013-1ce6-4c0d-9d8e-c60a81987c41
# ╠═24f463ee-76e6-11ef-0b34-796df7c80881
# ╟─62790b6a-c264-44f0-aeca-8116bd9a8471
# ╠═0fd3fb14-34f4-419b-bfe4-9eff238cc439
# ╠═18e316a8-9ec6-4343-8cf4-7a04eccd0df8
# ╟─40be7a21-175b-4032-af72-3b2258117402
# ╠═51d5b05d-683c-4d96-bc16-a6ef972f0d04
# ╠═74af4aed-0e24-4045-90ca-aa14a0284a44
# ╠═4dab33d9-3b14-4615-88a5-11fa0239ba65
# ╠═b67c80d7-be13-4e79-a14b-9aa6fea7eb78
# ╠═a6b8a27b-614c-42ba-941e-792f586b2881
# ╠═18262d61-d4d2-457e-a79a-77c3a365042c
# ╠═88fa84cd-b785-4bf7-841e-3c623c381c86
# ╠═2b027eda-4ec5-4315-bc7e-5a7774abac9b
# ╠═6794a588-7c72-47dd-8395-164545b6132a
# ╠═3835fef4-ab2a-44bf-a3b5-0bce1a70c605
# ╠═0050899c-2616-4f7f-aac7-c0a2eefb6ca2
# ╠═36c193e0-7531-4120-91e0-3e0ecdcb65bc
# ╠═0d12ebec-75c3-4f15-8098-9867bb6a3626
# ╠═b81de22d-469a-41e4-acf4-538d395bc653
# ╠═4f58a7ac-ae6e-4e43-9d59-8a5d1eab6554
# ╠═9982c4a7-bd44-4f04-a241-2eeeefaaa0fb
# ╠═a59ce994-3c51-4613-ac10-7fda447ef3af
