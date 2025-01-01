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

# ╔═╡ b0a42020-92ef-433c-b26b-36d3a82e5180
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 9b113c9b-a65b-49f3-abb8-582ff582e758
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, Statistics

# ╔═╡ 080f426e-6e15-4ae3-ae1c-43feee6b1788
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ e3a985b4-a755-4a55-8cf3-b031bae9f4e1
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 5d8f45e0-2f1d-43b7-ab3a-ffb1d1a4311c
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 15698332-14cd-479c-9345-836cd54c0965
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ dcf1f561-1d3b-4c73-a956-03f0ff720057
CACHEDIR = joinpath(@__DIR__, "cache_files", "KSS_Pavia")

# ╔═╡ b3d1a8d1-15d3-4509-a272-99fb854bc0b8
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 75bb6c2e-69f8-43b5-b29e-7f60bf698165
vars = matread(filepath)

# ╔═╡ e0c4e361-535d-4952-9173-6e9f0b24a8b1
vars_gt = matread(gt_filepath)

# ╔═╡ 2e7cb89c-08e5-4ae1-9fe7-9e95a3421ec9
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ c4dc77d4-99c9-4211-b981-b033b5b288f2
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ bc2ed5ae-17cd-46c1-940c-b5794c25f9ff
data = vars[data_key]

# ╔═╡ b1da7a62-8c57-420b-b580-08ba6e314929
gt_data = vars_gt[gt_key]

# ╔═╡ 1d00d4ad-ff49-47be-a6e5-8387a2bc4398
gt_labels = sort(unique(gt_data))

# ╔═╡ f161f22e-d287-4381-9b22-21e44c12c787
bg_indices = findall(gt_data .== 0)

# ╔═╡ 19f3bc12-bc98-4fa5-b3cb-6c12230618eb
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ b6624b05-92f7-4790-ab36-bd8387a2a7a2
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 36651253-c082-4f26-b47a-9ec24f8af906
n_classes = length(unique(gt_data)) - 1

# ╔═╡ 9ee29df1-b24b-4c24-83e4-2d89b4811aed
md"""
### Heatmap for Ground Truth
"""

# ╔═╡ 5e6d387c-6fb6-4bee-8ea9-8fe520c98c99
with_theme() do
	fig = Figure(; size=(500, 600))
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)
	fig
end

# ╔═╡ 5c15ab9d-c704-4fa2-acd4-c5cf0dfaa37b
data_mat = permutedims(reshape(data, :, size(data, 3)))

# ╔═╡ d3f08059-b747-4fc5-83d7-a72f8ca8553f
gt_vec = vec(gt_data)

# ╔═╡ ca8f1f61-ca93-4b5a-9fe1-69cba3d19a9d
md"""
**Mean Classifier**
"""

# ╔═╡ 1ca72f73-5d65-48de-b888-3884446995bd
struct MeanClassifier{T<:AbstractFloat}
	U::Vector{Vector{T}}
end

# ╔═╡ 0da4baee-1636-435d-b27d-a7c606936b22
function fit_mean(data::Array{T, 3}, gt_data::Array{U, 2}) where {T, U}

	#reshape the cube to a matrix
	data_mat = permutedims(reshape(data, :, size(data, 3)))

	#reshape the label matrix to a vector
	gt_vec = vec(gt_data)
	gt_labels = sort(unique(gt_vec))

	#filter out the unique classes from the background labels
	valid_labels = gt_labels

	#Initialize a list to store mean vectors
	mean_vectors = Vector{Vector{T}}()

	for label in valid_labels

		# Find indices of pixels belonging to the current label
		indices = findall(x -> x .== label, gt_vec)
		
		# Select 100 random indices for the current label
		selected_indices = indices[rand(1:length(indices), min(100, length(indices)))]
		
		# Select corresponding data points from data mat
		selected_points = data_mat[:, selected_indices]

		#Compute mean vector for the current label
		mean_vector = mean(selected_points, dims=2) |> vec

		push!(mean_vectors, mean_vector)
	end

	return MeanClassifier{T}(mean_vectors)
end

# ╔═╡ 47e0b117-bf16-4978-af8c-456f4cef0fd9
mean_classifier = fit_mean(data, gt_data)

# ╔═╡ b53bc942-6ddd-4c5b-8770-f6bc0f481c1f
Dnorms = dropdims(sum(abs2, data; dims=3); dims=3)

# ╔═╡ cc800d15-7bde-4220-9ecb-f4ab1d64d216
sum(abs2, data; dims=3)

# ╔═╡ 234c5f6e-1972-48ed-b5f5-151211f13819
function classify(sample::Array{T, 3}, classifier::MeanClassifier{T}) where T<:AbstractFloat
	Dnorms = dropdims(sum(abs2, sample; dims=3); dims=3)
	resids = map(classifier.U) do Uc
		innerprod = reshape(reshape(sample, :, size(sample,3))*Uc, size(sample)[1:2])
		sqrt.(Dnorms .- 2 .* innerprod .+ sum(abs2, Uc))
	end
	pixel_classes = map(CartesianIndices(size(sample)[1:2])) do idx
		min_idx = argmin(resid[idx] for resid in resids) - 1
		
	end
	# predicted_class = mode(vec(pixel_classes))
	return pixel_classes
end

# ╔═╡ 7e96728d-6e00-4a61-99a2-a3a21e868618
# classify(data, mean_classifier)

# ╔═╡ 83478be9-eeba-43c8-95c4-d542fb769bff
# unique(classify(data, mean_classifier))

# ╔═╡ 200bea11-dde0-4c49-bc07-5ad54387aef0
# with_theme() do
# 	fig = Figure(; size=(500, 600))
# 	colors = Makie.Colors.distinguishable_colors(n_classes + 1)
# 	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
# 	hm = heatmap!(ax, permutedims(classify(data, mean_classifier)); colormap=Makie.Categorical(colors), colorrange=(0, 9))
# 	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)
# 	fig
# end

# ╔═╡ 3b89c4a3-724b-41a4-b850-ceeffb776996
md"
### Rough Work
"

# ╔═╡ c4f03beb-27f1-428b-a87b-f191fed9ca41
indices_1 = findall(x -> x .== 6, gt_vec)

# ╔═╡ 16827b02-cb47-4cfe-9615-8e94298f204c
gt_vec[598654]

# ╔═╡ 9ffebe00-e44e-4345-a464-a10036795f28
selc_indices = indices_1[rand(1:length(indices_1), min(20, length(indices_1)))]

# ╔═╡ 286551db-d32d-461b-980c-565d395874b1
sel_data = data_mat[:, selc_indices]

# ╔═╡ af281f18-296a-458d-87ec-25add69c49fa
valid_labels = gt_labels[gt_labels .!= 0]

# ╔═╡ 002fff22-fdd4-45d8-b484-f144fc0752d6
mean_vectors = Vector{Vector{Float64}}()

# ╔═╡ 49c40878-f6b0-4669-9fc0-61dcacd53358
for label in valid_labels
	
	indices = findall(x -> x .== label, gt_vec)

	rand_indx = indices[rand(1:length(indices), min(100, length(indices)))]
	sel_data = data_mat[:, rand_indx]
	mean_vecs = mean(sel_data, dims=2) |> vec
	push!(mean_vectors, mean_vecs)
end

# ╔═╡ 192ff092-7d1f-4ca0-8cf6-ec673f4faffa
mean_vectors

# ╔═╡ Cell order:
# ╟─080f426e-6e15-4ae3-ae1c-43feee6b1788
# ╠═b0a42020-92ef-433c-b26b-36d3a82e5180
# ╠═9b113c9b-a65b-49f3-abb8-582ff582e758
# ╠═e3a985b4-a755-4a55-8cf3-b031bae9f4e1
# ╠═5d8f45e0-2f1d-43b7-ab3a-ffb1d1a4311c
# ╠═15698332-14cd-479c-9345-836cd54c0965
# ╠═dcf1f561-1d3b-4c73-a956-03f0ff720057
# ╠═b3d1a8d1-15d3-4509-a272-99fb854bc0b8
# ╠═75bb6c2e-69f8-43b5-b29e-7f60bf698165
# ╠═e0c4e361-535d-4952-9173-6e9f0b24a8b1
# ╠═2e7cb89c-08e5-4ae1-9fe7-9e95a3421ec9
# ╠═c4dc77d4-99c9-4211-b981-b033b5b288f2
# ╠═bc2ed5ae-17cd-46c1-940c-b5794c25f9ff
# ╠═b1da7a62-8c57-420b-b580-08ba6e314929
# ╠═1d00d4ad-ff49-47be-a6e5-8387a2bc4398
# ╠═f161f22e-d287-4381-9b22-21e44c12c787
# ╟─19f3bc12-bc98-4fa5-b3cb-6c12230618eb
# ╠═b6624b05-92f7-4790-ab36-bd8387a2a7a2
# ╠═36651253-c082-4f26-b47a-9ec24f8af906
# ╟─9ee29df1-b24b-4c24-83e4-2d89b4811aed
# ╠═5e6d387c-6fb6-4bee-8ea9-8fe520c98c99
# ╠═5c15ab9d-c704-4fa2-acd4-c5cf0dfaa37b
# ╠═d3f08059-b747-4fc5-83d7-a72f8ca8553f
# ╟─ca8f1f61-ca93-4b5a-9fe1-69cba3d19a9d
# ╠═1ca72f73-5d65-48de-b888-3884446995bd
# ╠═0da4baee-1636-435d-b27d-a7c606936b22
# ╠═47e0b117-bf16-4978-af8c-456f4cef0fd9
# ╠═b53bc942-6ddd-4c5b-8770-f6bc0f481c1f
# ╠═cc800d15-7bde-4220-9ecb-f4ab1d64d216
# ╠═234c5f6e-1972-48ed-b5f5-151211f13819
# ╠═7e96728d-6e00-4a61-99a2-a3a21e868618
# ╠═83478be9-eeba-43c8-95c4-d542fb769bff
# ╠═200bea11-dde0-4c49-bc07-5ad54387aef0
# ╟─3b89c4a3-724b-41a4-b850-ceeffb776996
# ╠═c4f03beb-27f1-428b-a87b-f191fed9ca41
# ╠═16827b02-cb47-4cfe-9615-8e94298f204c
# ╠═9ffebe00-e44e-4345-a464-a10036795f28
# ╠═286551db-d32d-461b-980c-565d395874b1
# ╠═af281f18-296a-458d-87ec-25add69c49fa
# ╠═002fff22-fdd4-45d8-b484-f144fc0752d6
# ╠═49c40878-f6b0-4669-9fc0-61dcacd53358
# ╠═192ff092-7d1f-4ca0-8cf6-ec673f4faffa
