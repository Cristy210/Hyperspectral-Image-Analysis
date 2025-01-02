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

# ╔═╡ 11df6eb6-798c-48b2-b235-2e0668e86722
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 95cf0240-a2bd-455e-8752-5c35a80408cc
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, Statistics

# ╔═╡ 91391c39-2b54-474b-90a9-22b5c67b40e2
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 72c19f2d-642b-49ae-ba28-4a20aa2bd9e2
filepath = joinpath(@__DIR__, "MAT Files", "Salinas_corrected.mat")

# ╔═╡ e9b5d781-29bf-4f50-bd76-ce62a9520688
gt_filepath = joinpath(@__DIR__, "GT Files", "Salinas_gt.mat")

# ╔═╡ c59016c2-3a33-468f-95d6-b1656536898e
CACHEDIR = joinpath(@__DIR__, "cache_files", "KSS_Salinas")

# ╔═╡ 2905dc6e-ba02-416b-9769-723a0aae1509
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ f92f7f85-6c92-4e4b-8890-ea6038251c18
vars = matread(filepath)

# ╔═╡ 90ef5e62-5f6d-4fa9-8dc4-06bdb3bea8dd
vars_gt = matread(gt_filepath)

# ╔═╡ ba25f5b8-0cc3-4e2d-b1b8-1b5668de20c4
data = vars["salinas_corrected"]

# ╔═╡ a3b059df-93ad-4526-af23-0b17d94c8709
gt_data = vars_gt["salinas_gt"]

# ╔═╡ 6cebf8fd-0fd0-435c-bef0-95838a063b84
gt_labels = sort(unique(gt_data))

# ╔═╡ 2f7ff7c6-f75e-4a81-b40d-5d0deccb5b6d
bg_indices = findall(gt_data .== 0)

# ╔═╡ 1bca6dca-a336-4993-acb2-7f12af9ac07f
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 6baf8f26-bab6-41ea-8ea6-6188556459d2
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 90e24d33-d79c-4bfc-b70b-7dfa9e8e4d09
n_classes = length(unique(gt_data)) - 1

# ╔═╡ 53168f41-05bc-444e-9860-77e2c2179002
md"""
## Mean Classifier
"""

# ╔═╡ 48c08eb6-cc53-49ba-9c53-e7ee4843649f
struct MeanClassifier{T<:AbstractFloat}
	U::Vector{Vector{T}}
end

# ╔═╡ 9b941cce-34e1-4081-ae02-482d80c91d98
function fit_mean(data::Array{T, 3}, gt_data::Array{U, 2}) where {T, U}

	#reshape the cube to a matrix
	data_mat = permutedims(reshape(data, :, size(data, 3)))

	#reshape the label matrix to a vector
	gt_vec = vec(gt_data)
	gt_labels = sort(unique(gt_vec))

	#filter out the unique classes from the background labels
	valid_labels = gt_labels[gt_labels .!= 0]

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

# ╔═╡ 3c39ec8e-a110-4dc0-b111-b09dc62a4489
mean_classifier = fit_mean(data, gt_data)

# ╔═╡ b9080207-017e-4bf0-a3a5-9d390d2bb0bf
function classify(sample::Array{T, 2}, classifier::MeanClassifier{T}) where T<:AbstractFloat
	Dnorms = dropdims(sum(abs2, sample; dims=1); dims=1)
	resids = map(classifier.U) do Uc
		innerprod = permutedims(sample)*Uc
		sqrt.(Dnorms .- 2 .* innerprod .+ sum(abs2, Uc))
	end
	pixel_classes = map(1:size(sample)[2]) do idx
		min_idx = argmin(resid[idx] for resid in resids)
		
	end
	# predicted_class = mode(vec(pixel_classes))
	return pixel_classes
end

# ╔═╡ 2c122e5a-b5f8-4086-a414-3bb565bcda16
md"
## Ground Truth Vs Mean Classification
"

# ╔═╡ 758b5863-4ed0-4b1d-b486-62c63ff68749
md"
## Confusion Matrix - Mean Classification
"

# ╔═╡ a9d60506-9ca8-4a61-903e-d5be77a818a0
md"
## Subspace Classifier
"

# ╔═╡ a486d58a-f715-4631-b401-e91858b01ca0
struct SubspaceClassifier{T<:AbstractFloat}
	U::Vector{Matrix{T}}
	thresh::Vector{T}
end

# ╔═╡ 60365a6e-c92b-480b-bef5-c9de10a71f71
gt_vec = vec(gt_data)

# ╔═╡ c746780e-f0d2-4160-ac9a-f54636119d21
valid_labels = Int64.(gt_labels[gt_labels .!= 0])

# ╔═╡ 504a0a92-f5d2-4d11-89fa-c948cf74f2af
function fit_subspace(data::Array{T, 3}, gt_data::Array{U, 2}, dims::Vector{Int}, thresh::Vector{T}) where {T, U}
	
	#reshape the cube to a matrix
	data_mat = permutedims(reshape(data, :, size(data, 3)))

	#reshape the label matrix to a vector
	gt_vec = vec(gt_data)
	gt_labels = sort(unique(gt_vec))

	#filter out the unique classes from the background labels
	valid_labels = Int64.(gt_labels[gt_labels .!= 0])

	#Initialize a list to store subspace basis
	subspace_basis = Vector{Matrix{T}}()

	for label in valid_labels

		# Find indices of pixels belonging to the current label
		indices = findall(x -> x .== label, gt_vec)
		
		# Select 100 random indices for the current label
		selected_indices = indices[rand(1:length(indices), min(100, length(indices)))]
		
		# Select corresponding data points from data mat
		selected_points = data_mat[:, selected_indices]

		#Compute Subspace basis for the current label
		U_label = svd(selected_points).U[:, 1:dims[label]]

		push!(subspace_basis, U_label)
	end
	
	
	return SubspaceClassifier{T}(subspace_basis, thresh)
end

# ╔═╡ 9f7e0fc8-a8a1-448d-b787-e480e4a5e0aa
md"
### Selected the desired dimensions for each class
"

# ╔═╡ 347393a7-e024-4305-88e5-5c903ed557ba
@bind Dimensions Select(["1", "2", "3"])

# ╔═╡ 3210452d-4372-4edd-be01-8a90a47c8d8f
Select_Dims = Dict(["1" => fill(1, n_classes), "2" => fill(2, n_classes), "3" => fill(3, n_classes)])

# ╔═╡ 20825110-5277-4145-ade4-64d9cdfad5bd
n_dims = Select_Dims[Dimensions]

# ╔═╡ af75265f-beaf-4577-9aca-729a221f03d8
Threshold = fill(Inf64, n_classes)

# ╔═╡ f5a105cd-f3ae-410e-860f-4ed317cd745d
subspace_classifier = fit_subspace(data, gt_data, n_dims, Threshold)

# ╔═╡ fbf7b12e-cbc1-4e96-b5d0-dd0658793a12
function classify(sample::Array{T, 2}, classifier::SubspaceClassifier{T}) where T<:AbstractFloat
	Dnorms = dropdims(sum(abs2, sample; dims=1); dims=1)
	resids = map(classifier.U) do Uc
		UDnorms = dropdims(sum(abs2, Uc'sample; dims=1); dims=1)
		Dnorms .- UDnorms
	end
	pixel_classes = map(1:size(sample)[2]) do idx
		min_idx = argmin(resid[idx] for resid in resids)
		if resids[min_idx][idx] < classifier.thresh[min_idx]
			return min_idx
		else
			return nothing
		end
	end
	# predicted_class = mode(vec(pixel_classes))
	return pixel_classes
end

# ╔═╡ e29055bf-099e-4258-a199-e01f48ecb097
labels_mean = classify(permutedims(data[mask, :]), mean_classifier)

# ╔═╡ 73b473b5-812c-48e5-b5cc-cca557d93730
with_theme() do
	fig = Figure(; size=(800, 650))
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= labels_mean
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)

	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2,1], hm, tellwidth=false,vertical=false, ticklabelsize=:8)

	
	ax = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true, title="Mean Classification", titlesize=15)
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2, 2], hm, tellwidth=false,vertical=false, ticklabelsize=:8)
	fig
end

# ╔═╡ 271876b2-7a35-4cce-a133-831b70fb393d
begin
	ground_labels_mean = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels_mean = length(ground_labels_mean)
	predicted_labels_mean = n_classes

	confusion_matrix_mean = zeros(Float64, true_labels_mean, predicted_labels_mean) #Initialize a confusion matrix filled with zeros
	cluster_results_mean = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	# clu_assign, idx = spec_aligned, spec_clustering_idx
	

	cluster_results_mean[mask] .= labels_mean

	for (label_idx, label) in enumerate(ground_labels_mean)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results_mean[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_classes]
		confusion_matrix_mean[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ 22c64a82-f985-4330-acf4-0ddd5f2cf98a
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_mean, yticks = 1:true_labels_mean, title="Confusion Matrix - Mean Classification - Salinas")
	hm = heatmap!(ax, permutedims(confusion_matrix_mean), colormap=:viridis)
	pm = permutedims(confusion_matrix_mean)

	for i in 1:true_labels_mean, j in 1:predicted_labels_mean
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 8b1fc2f2-2cd9-48ba-b6c4-9e2f88e582c5
unique(classify(permutedims(data[mask, :]), mean_classifier))

# ╔═╡ 01a0f598-0a1c-419e-9085-8405014e67f9
labels_subspace = classify(permutedims(data[mask, :]), subspace_classifier)

# ╔═╡ e33d4247-a59f-40ef-9627-3d1ca268ffc9
with_theme() do
	fig = Figure(; size=(800, 650))
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= labels_subspace
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)

	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	
	ax = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true, title="Subspace Classification - Dims = $Dimensions", titlesize=15)
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2, 2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)
	fig
end

# ╔═╡ a95e8c22-fbd9-4625-8991-69b482fbd902
md"
## Affine Classifier
"

# ╔═╡ c33aeeb4-e671-429a-a38a-ea8122e933e1


# ╔═╡ Cell order:
# ╟─91391c39-2b54-474b-90a9-22b5c67b40e2
# ╠═11df6eb6-798c-48b2-b235-2e0668e86722
# ╠═95cf0240-a2bd-455e-8752-5c35a80408cc
# ╠═72c19f2d-642b-49ae-ba28-4a20aa2bd9e2
# ╠═e9b5d781-29bf-4f50-bd76-ce62a9520688
# ╠═c59016c2-3a33-468f-95d6-b1656536898e
# ╠═2905dc6e-ba02-416b-9769-723a0aae1509
# ╠═f92f7f85-6c92-4e4b-8890-ea6038251c18
# ╠═90ef5e62-5f6d-4fa9-8dc4-06bdb3bea8dd
# ╠═ba25f5b8-0cc3-4e2d-b1b8-1b5668de20c4
# ╠═a3b059df-93ad-4526-af23-0b17d94c8709
# ╠═6cebf8fd-0fd0-435c-bef0-95838a063b84
# ╠═2f7ff7c6-f75e-4a81-b40d-5d0deccb5b6d
# ╟─1bca6dca-a336-4993-acb2-7f12af9ac07f
# ╠═6baf8f26-bab6-41ea-8ea6-6188556459d2
# ╠═90e24d33-d79c-4bfc-b70b-7dfa9e8e4d09
# ╟─53168f41-05bc-444e-9860-77e2c2179002
# ╠═48c08eb6-cc53-49ba-9c53-e7ee4843649f
# ╠═9b941cce-34e1-4081-ae02-482d80c91d98
# ╠═3c39ec8e-a110-4dc0-b111-b09dc62a4489
# ╠═b9080207-017e-4bf0-a3a5-9d390d2bb0bf
# ╠═e29055bf-099e-4258-a199-e01f48ecb097
# ╠═8b1fc2f2-2cd9-48ba-b6c4-9e2f88e582c5
# ╟─2c122e5a-b5f8-4086-a414-3bb565bcda16
# ╠═73b473b5-812c-48e5-b5cc-cca557d93730
# ╟─758b5863-4ed0-4b1d-b486-62c63ff68749
# ╠═271876b2-7a35-4cce-a133-831b70fb393d
# ╠═22c64a82-f985-4330-acf4-0ddd5f2cf98a
# ╟─a9d60506-9ca8-4a61-903e-d5be77a818a0
# ╠═a486d58a-f715-4631-b401-e91858b01ca0
# ╠═60365a6e-c92b-480b-bef5-c9de10a71f71
# ╠═c746780e-f0d2-4160-ac9a-f54636119d21
# ╠═504a0a92-f5d2-4d11-89fa-c948cf74f2af
# ╟─9f7e0fc8-a8a1-448d-b787-e480e4a5e0aa
# ╠═347393a7-e024-4305-88e5-5c903ed557ba
# ╠═3210452d-4372-4edd-be01-8a90a47c8d8f
# ╠═20825110-5277-4145-ade4-64d9cdfad5bd
# ╠═af75265f-beaf-4577-9aca-729a221f03d8
# ╠═f5a105cd-f3ae-410e-860f-4ed317cd745d
# ╠═fbf7b12e-cbc1-4e96-b5d0-dd0658793a12
# ╠═01a0f598-0a1c-419e-9085-8405014e67f9
# ╠═e33d4247-a59f-40ef-9627-3d1ca268ffc9
# ╟─a95e8c22-fbd9-4625-8991-69b482fbd902
# ╠═c33aeeb4-e671-429a-a38a-ea8122e933e1
