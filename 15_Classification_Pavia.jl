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
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ b5bc9bac-9c23-41b4-8730-696a6ea196e0
md"""
### Activate Project Directory
"""

# ╔═╡ afa8c064-dd3f-445e-96f4-84b3c86baf1a
md"""
### Install Necessary Packages
"""

# ╔═╡ 3b596463-35c4-4b12-b663-401af311dc02
md"""
### Select the Dataset
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

# ╔═╡ ca8f1f61-ca93-4b5a-9fe1-69cba3d19a9d
md"""
## Mean Classifier
"""

# ╔═╡ 1ca72f73-5d65-48de-b888-3884446995bd
struct MeanClassifier{T<:AbstractFloat}
	b::Vector{Vector{T}}
end

# ╔═╡ 0da4baee-1636-435d-b27d-a7c606936b22
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
		
		# Select 3000 random indices for the current label
		selected_indices = indices[rand(1:length(indices), min(3000, length(indices)))]
		
		# Select corresponding data points from data mat
		selected_points = data_mat[:, selected_indices]

		#Compute mean vector for the current label
		mean_vector = mean(selected_points, dims=2) |> vec

		push!(mean_vectors, mean_vector)
	end

	return MeanClassifier{T}(mean_vectors)
end

# ╔═╡ c0a17298-c27e-4593-aebc-273576ff9053
dropdims(sum(abs2, permutedims(data[mask, :]); dims=1); dims=1)

# ╔═╡ 284327c2-726b-4f5d-aabf-e6cf22c9094f
permutedims(data[mask, :])

# ╔═╡ 47e0b117-bf16-4978-af8c-456f4cef0fd9
mean_classifier = fit_mean(data, gt_data)

# ╔═╡ 234c5f6e-1972-48ed-b5f5-151211f13819
function classify(sample::Array{T, 2}, classifier::MeanClassifier{T}) where T<:AbstractFloat
	Dnorms = dropdims(sum(abs2, sample; dims=1); dims=1)
	resids = map(classifier.b) do Uc
		innerprod = permutedims(sample)*Uc
		sqrt.(Dnorms .- 2 .* innerprod .+ sum(abs2, Uc))
	end
	pixel_classes = map(1:size(sample)[2]) do idx
		min_idx = argmin(resid[idx] for resid in resids)
		
	end
	# predicted_class = mode(vec(pixel_classes))
	return pixel_classes
end

# ╔═╡ d64f5ea0-f95d-4b0f-803a-e3fca082006d
# unique(classify(permutedims(data[mask, :]), mean_classifier))

# ╔═╡ 26e3f3ea-9bcb-44df-966b-4506d1fe548a
md"
## Ground Truth Vs Mean Classification
"

# ╔═╡ d25e33e2-e2dd-46a5-b84a-a88aa92b7de5
md"
## Confusion Matrix - Mean Classification
"

# ╔═╡ e428a263-d609-413d-97e8-c7a6981dde8e
md"
## Subspace Classifier
"

# ╔═╡ 88b1ad11-f28c-45ec-b3eb-b301cc5447f3
struct SubspaceClassifier{T<:AbstractFloat}
	U::Vector{Matrix{T}}
	thresh::Vector{T}
end

# ╔═╡ 51266b17-94ce-4c6f-bda6-f965f84fcff5
function fit_subspace(data::Array{T, 3}, gt_data::Array{U, 2}, dims::Vector{Int}, thresh::Vector{T}) where {T, U}
	
	#reshape the cube to a matrix
	data_mat = permutedims(reshape(data, :, size(data, 3)))

	#reshape the label matrix to a vector
	gt_vec = vec(gt_data)
	gt_labels = sort(unique(gt_vec))

	#filter out the unique classes from the background labels
	valid_labels = gt_labels[gt_labels .!= 0]

	#Initialize a list to store subspace basis
	subspace_basis = Vector{Matrix{T}}()

	for label in valid_labels

		# Find indices of pixels belonging to the current label
		indices = findall(x -> x .== label, gt_vec)
		
		# Select 3000 random indices for the current label
		selected_indices = indices[rand(1:length(indices), min(3000, length(indices)))]
		
		# Select corresponding data points from data mat
		selected_points = data_mat[:, selected_indices]

		#Compute Subspace basis for the current label
		U_label = svd(selected_points).U[:, 1:dims[label]]

		push!(subspace_basis, U_label)
	end
	
	
	return SubspaceClassifier{T}(subspace_basis, thresh)
end

# ╔═╡ 22e11b97-3b91-4bd9-863b-8c205cfc6d9f
md"
### Selected the desired dimensions for each class
"

# ╔═╡ 57289efe-7de4-4daa-b5dc-213a57a3d084
@bind Dimensions Select(["1", "2", "4", "8", "16", "32"])

# ╔═╡ a36e5092-5d00-4085-86bb-d7aef02cfb76
Select_Dims = Dict(["1" => fill(1, n_classes), "2" => fill(2, n_classes), "4" => fill(4, n_classes), "8" => fill(8, n_classes), "16" => fill(16, n_classes), "32" => fill(32, n_classes)])

# ╔═╡ 1b48dfb1-d361-44e2-83d1-5264b23998b7
n_dims = Select_Dims[Dimensions]

# ╔═╡ 6382ac76-5d26-4710-94c2-5bfc1ba22597
Threshold = fill(Inf64, n_classes)

# ╔═╡ 320d30a2-b612-4cdc-aae5-04d9384314aa
subspace_classifier = fit_subspace(data, gt_data, n_dims, Threshold)

# ╔═╡ 0c268def-9af8-480c-9f8a-2a7adb6c56f8
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

# ╔═╡ 1c703eee-2e38-46d3-a97f-623ebd824b84
md"
## Ground Truth Vs Subspace Classification
"

# ╔═╡ 2a580d27-dce1-4da7-bc62-12aeb46330e9
md"
## Confusion Matrix - Subspace Classification
"

# ╔═╡ 9a3e6c31-af27-44c9-ac92-bec97391579c
md"
## Affine Classifier
"

# ╔═╡ db2fb8a1-93c8-4d20-a942-68b04a86c693
struct AffineClassifier{T<:AbstractFloat}
	b::Vector{Vector{T}}
	U::Vector{Matrix{T}}
	thresh::Vector{T}
end

# ╔═╡ 15a4f896-277b-4df0-ba1d-9dae1c9d0e92
function fit_affinespace(data::Array{T, 3}, gt_data::Array{U, 2}, dims::Vector{Int}, thresh::Vector{T}) where {T, U}
	
	#reshape the cube to a matrix
	data_mat = permutedims(reshape(data, :, size(data, 3)))

	#reshape the label matrix to a vector
	gt_vec = vec(gt_data)
	gt_labels = sort(unique(gt_vec))

	#filter out the unique classes from the background labels
	valid_labels = Int64.(gt_labels[gt_labels .!= 0])

	#Initialize a list to store subspace basis and mean vector
	subspace_basis = Vector{Matrix{T}}()
	mean_vectors = Vector{Vector{T}}()

	for label in valid_labels

		# Find indices of pixels belonging to the current label
		indices = findall(x -> x .== label, gt_vec)

		# Select 3000 random indices for the current label
		selected_indices = indices[rand(1:length(indices), min(3000, length(indices)))]

		# Select corresponding data points from data mat
		selected_points = data_mat[:, selected_indices]

		#Compute mean vector for the current label
		mean_vector = mean(selected_points, dims=2) |> vec
		push!(mean_vectors, mean_vector)

		#Compute Subspace basis for the current label
		U_label = svd(selected_points).U[:, 1:dims[label]]
		push!(subspace_basis, U_label)

	end

	return AffineClassifier{T}(mean_vectors, subspace_basis, thresh)
end

# ╔═╡ 7c055c89-0979-48eb-a981-3c0c2ed3a0f8
affine_classifier = fit_affinespace(data, gt_data, n_dims, Threshold)

# ╔═╡ 75e824e1-1cbc-44f0-b211-2ac76abfc68b
function classify(sample::Array{T, 2}, classifier::AffineClassifier{T}) where T<:AbstractFloat
	resids = map(zip(classifier.U, classifier.b)) do (Uhat, bhat)
		dropdims(sum(abs2, sample .- (Uhat*Uhat'*(sample.- bhat) .+ bhat); dims=1); dims=1)
	end

	pixel_classes = map(1:size(sample)[2]) do idx
		min_idx = argmin(resid[idx] for resid in resids)
		if resids[min_idx][idx] < classifier.thresh[min_idx]
			return min_idx
		else
			return nothing
		end
	end
	return pixel_classes
end

# ╔═╡ dff407e5-17d3-433f-9f9f-779f89d4f5d1
labels_mean = classify(permutedims(data[mask, :]), mean_classifier)

# ╔═╡ 200bea11-dde0-4c49-bc07-5ad54387aef0
with_theme() do
	fig = Figure(; size=(800, 650))
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= labels_mean
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)

	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	
	ax = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true, title="Mean Classification", titlesize=15)
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2, 2], hm, tellwidth=false, vertical=false)
	fig
end

# ╔═╡ 23ee3fa2-bbe8-479f-95b1-423578f7e030
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

# ╔═╡ 2b53268b-4d1a-471c-addd-783eaa813c55
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_mean, yticks = 1:true_labels_mean, title="Confusion Matrix - Mean Classification - $Location")
	hm = heatmap!(ax, permutedims(confusion_matrix_mean), colormap=:viridis)
	pm = permutedims(confusion_matrix_mean)

	for i in 1:true_labels_mean, j in 1:predicted_labels_mean
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=15)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ dc6df293-74b5-41a0-befe-6a0aa9fcc130
labels_subspace = classify(permutedims(data[mask, :]), subspace_classifier)

# ╔═╡ 26d64190-8748-4673-87ef-833b15f284d0
with_theme() do
	fig = Figure(; size=(800, 650))
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= labels_subspace
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)

	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	
	ax = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true, title="Subspace - Dims = $Dimensions", titlesize=15)
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2, 2], hm, tellwidth=false, vertical=false)
	fig
end

# ╔═╡ e42f949a-d22b-44ff-b1fb-385bcead2b40
begin
	ground_labels_re = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels_re = length(ground_labels_re)
	predicted_labels_re = n_classes

	confusion_matrix_subspace = zeros(Float64, true_labels_re, predicted_labels_re) #Initialize a confusion matrix filled with zeros
	cluster_results_re = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	# clu_assign, idx = spec_aligned, spec_clustering_idx
	

	cluster_results_re[mask] .= labels_subspace

	for (label_idx, label) in enumerate(ground_labels_re)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results_re[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_classes]
		confusion_matrix_subspace[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ 87f7368b-a0c9-4eac-b95f-0109e5c82023
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Subspace Classification - Dims = $Dimensions - $Location")
	hm = heatmap!(ax, permutedims(confusion_matrix_subspace), colormap=:viridis)
	pm = permutedims(confusion_matrix_subspace)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=15)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ c0da6a0e-c4e1-418c-86d2-9c1be6e3131a
labels_affine = classify(permutedims(data[mask, :]), affine_classifier)

# ╔═╡ 61a6c794-67c7-4152-979f-53a59cf3f3d4
md"
## Ground Truth Vs Affine space Classification
"

# ╔═╡ c295ca62-da1c-4fee-a6df-69923cd00ef6
with_theme() do
	fig = Figure(; size=(800, 650))
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= labels_affine
	colors = Makie.Colors.distinguishable_colors(n_classes + 1)

	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	
	ax = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true, title="Affine space - $Location - Dims = $Dimensions", titlesize=15)
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_classes))
	Colorbar(fig[2, 2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)
	fig
end

# ╔═╡ f0e862b0-6980-417c-988d-b125fa6387f2
md"
## Confusion Matrix - Affine Classification
"

# ╔═╡ 7f2931e9-7dcd-4c97-9f96-06d884ad2831
begin
	ground_labels_af = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels_af = length(ground_labels_af)
	predicted_labels_af = n_classes

	confusion_matrix_affine = zeros(Float64, true_labels_af, predicted_labels_af) #Initialize a confusion matrix filled with zeros
	cluster_results_af = fill(NaN32, size(data)[1:2]) #Clustering algorithm results
	

	cluster_results_af[mask] .= labels_affine

	for (label_idx, label) in enumerate(ground_labels_af)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results_af[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_classes]
		confusion_matrix_affine[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ ff6f02e4-5ca5-4cbf-8b7a-11ba3dc33642
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_af, yticks = 1:true_labels_af, title=" Affine Classification - Dims = $Dimensions - $Location")
	hm = heatmap!(ax, permutedims(confusion_matrix_affine), colormap=:viridis)
	pm = permutedims(confusion_matrix_affine)

	for i in 1:true_labels_af, j in 1:predicted_labels_af
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=15)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╟─080f426e-6e15-4ae3-ae1c-43feee6b1788
# ╟─b5bc9bac-9c23-41b4-8730-696a6ea196e0
# ╠═b0a42020-92ef-433c-b26b-36d3a82e5180
# ╟─afa8c064-dd3f-445e-96f4-84b3c86baf1a
# ╠═9b113c9b-a65b-49f3-abb8-582ff582e758
# ╟─3b596463-35c4-4b12-b663-401af311dc02
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
# ╟─ca8f1f61-ca93-4b5a-9fe1-69cba3d19a9d
# ╠═1ca72f73-5d65-48de-b888-3884446995bd
# ╠═0da4baee-1636-435d-b27d-a7c606936b22
# ╠═c0a17298-c27e-4593-aebc-273576ff9053
# ╠═284327c2-726b-4f5d-aabf-e6cf22c9094f
# ╠═47e0b117-bf16-4978-af8c-456f4cef0fd9
# ╠═234c5f6e-1972-48ed-b5f5-151211f13819
# ╠═dff407e5-17d3-433f-9f9f-779f89d4f5d1
# ╠═d64f5ea0-f95d-4b0f-803a-e3fca082006d
# ╟─26e3f3ea-9bcb-44df-966b-4506d1fe548a
# ╟─200bea11-dde0-4c49-bc07-5ad54387aef0
# ╟─d25e33e2-e2dd-46a5-b84a-a88aa92b7de5
# ╠═23ee3fa2-bbe8-479f-95b1-423578f7e030
# ╟─2b53268b-4d1a-471c-addd-783eaa813c55
# ╟─e428a263-d609-413d-97e8-c7a6981dde8e
# ╠═88b1ad11-f28c-45ec-b3eb-b301cc5447f3
# ╠═51266b17-94ce-4c6f-bda6-f965f84fcff5
# ╟─22e11b97-3b91-4bd9-863b-8c205cfc6d9f
# ╠═57289efe-7de4-4daa-b5dc-213a57a3d084
# ╠═a36e5092-5d00-4085-86bb-d7aef02cfb76
# ╠═1b48dfb1-d361-44e2-83d1-5264b23998b7
# ╠═6382ac76-5d26-4710-94c2-5bfc1ba22597
# ╠═320d30a2-b612-4cdc-aae5-04d9384314aa
# ╠═0c268def-9af8-480c-9f8a-2a7adb6c56f8
# ╠═dc6df293-74b5-41a0-befe-6a0aa9fcc130
# ╟─1c703eee-2e38-46d3-a97f-623ebd824b84
# ╠═26d64190-8748-4673-87ef-833b15f284d0
# ╟─2a580d27-dce1-4da7-bc62-12aeb46330e9
# ╠═e42f949a-d22b-44ff-b1fb-385bcead2b40
# ╟─87f7368b-a0c9-4eac-b95f-0109e5c82023
# ╟─9a3e6c31-af27-44c9-ac92-bec97391579c
# ╠═db2fb8a1-93c8-4d20-a942-68b04a86c693
# ╠═15a4f896-277b-4df0-ba1d-9dae1c9d0e92
# ╠═7c055c89-0979-48eb-a981-3c0c2ed3a0f8
# ╠═75e824e1-1cbc-44f0-b211-2ac76abfc68b
# ╠═c0da6a0e-c4e1-418c-86d2-9c1be6e3131a
# ╟─61a6c794-67c7-4152-979f-53a59cf3f3d4
# ╟─c295ca62-da1c-4fee-a6df-69923cd00ef6
# ╟─f0e862b0-6980-417c-988d-b125fa6387f2
# ╠═7f2931e9-7dcd-4c97-9f96-06d884ad2831
# ╟─ff6f02e4-5ca5-4cbf-8b7a-11ba3dc33642
