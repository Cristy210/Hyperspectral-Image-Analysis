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

# ╔═╡ d8a3a9ae-9e86-4b3d-88ec-b923691f4f97
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 2a46e3fd-09c5-437c-8ce5-865bab3c65c1
using LinearAlgebra, Dictionaries, MLDatasets, CairoMakie, Statistics, Clustering, ProgressLogging, Logging, Random, MAT, PlutoUI

# ╔═╡ 22335c30-b725-11ef-2f34-6f691acbde73
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

# ╔═╡ 72b7dc20-271d-4821-bfbe-410fe1b29f28
function affine_approx(X, k)
	bhat = mean(eachcol(X))
	Uhat = svd(X - bhat*ones(size(X,2))').U[:,1:k]
	return bhat, Uhat
end

# ╔═╡ e4d94be6-0027-4df9-bb37-b36dbe64ffa6
md"""
## MNIST Data
"""

# ╔═╡ 7027a398-91f7-4d5f-860e-8e4242618b80
begin
	extract_images(data) = (dictionary ∘ map)(0:9) do digit
	image_cube = data.features[:, :, data.targets .== digit]
	image_mat = reshape(image_cube, :, size(image_cube, 3))
	return digit => image_mat
	end
	train = extract_images(MNIST(:train))
end

# ╔═╡ 4453e50e-0bf2-4cf3-a7e9-8ff5c3d034dc
CLASSIFIER_DIM = 1

# ╔═╡ 878f0a6a-43e1-4f23-9279-bedce534b576
fits = map(image_mat -> affine_approx(image_mat, CLASSIFIER_DIM), train)

# ╔═╡ 34f16752-e284-49ea-8335-16ba689ac924
md"""
###  Define classifier based on distance to affine subspaces
"""

# ╔═╡ d6e56291-7b34-45e5-b7b7-e789086942cb
function dist(x, (bhat, Uhat))
	return norm(x - (Uhat*Uhat'*(x-bhat) + bhat))
end

# ╔═╡ e66874cf-245d-4211-a9fd-76ab6733c04b
classify(x, fits) = argmin(i -> dist(x, fits[i]), 0:9)

# ╔═╡ 74b7e833-d38d-44f8-a248-cf0cfa9cac1a
begin
	test = extract_images(MNIST(:test))
	pred_counts = map(test) do image_mat
	preds = map(x -> classify(x, fits), eachcol(image_mat))
	counts = dictionary(pred => count(==(pred), preds) for pred in 0:9)
	end
end

# ╔═╡ a9813130-40f0-4aec-9b19-0ed5d869ddd6
num_correct = sum(pred_counts[class][class] for class in 0:9)

# ╔═╡ bbee969f-6527-4590-ae7c-51ab5f132521
prop_correct = num_correct / sum(sum.(collect.(pred_counts)))

# ╔═╡ 278f4db4-eb6b-4aed-91bc-1dce3d3f612b


# ╔═╡ 987d86be-434d-47f7-9416-934ab13cf729
md"""
## Pavia Dataset
"""

# ╔═╡ de552769-e5e7-4a28-8693-6b783517b223
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ fc01c02e-448c-43be-886d-788af2da8d26
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 7a83109f-8354-46b2-8391-331a8d8e9285
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ fc240a4f-b916-4e61-bb67-2ab45cfa533b
vars = matread(filepath)

# ╔═╡ f6533f60-a5f2-4bfc-a39a-2766e9ac3910
vars_gt = matread(gt_filepath)

# ╔═╡ fcafc53f-daaf-4831-a633-e5e2251ed02c
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 2c3a835b-cb6f-4e04-945e-9b46e4cf3aa7
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ bbfbf9fa-b0ff-4082-b5f7-ee70d7111f28
data = vars[data_key]

# ╔═╡ a38bb098-0907-49d5-9861-6ecec8be130b
gt_data = vars_gt[gt_key]

# ╔═╡ 7a0b949a-4ec0-4864-81d2-51828948b63c
gt_labels = sort(unique(gt_data))

# ╔═╡ 21e4f21f-4e9a-4c95-b4e8-a73c326c14ce
bg_indices = findall(gt_data .== 0)

# ╔═╡ d47100b1-6422-4585-9a2e-c2ef09fb9cf0
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 8451bec4-0ae1-4766-ae71-ba997617b6e2
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 5e0d9591-3f39-4ddf-a5be-c560af6def99
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 8f13a36f-d326-4b63-a87c-d2f642750eef
permutedims(data[mask, :])

# ╔═╡ 0e2268f3-18cd-4919-a90f-9e526589a8a7
training_labels = [1, 2, 3, 4]

# ╔═╡ 1e54d673-73c8-4389-aa49-6b031906900f
testing_labels = [5, 6, 7, 8, 9]

# ╔═╡ 1e8ad769-e259-4837-8cf8-b344ef994440
training_labels .∈ [1, 1, 1, 1]

# ╔═╡ 35ba8c5b-4de6-4ae7-8af3-d06753d32f7e
train_labels = gt_data[gt_data .∈ training_labels]

# ╔═╡ Cell order:
# ╟─22335c30-b725-11ef-2f34-6f691acbde73
# ╠═d8a3a9ae-9e86-4b3d-88ec-b923691f4f97
# ╠═2a46e3fd-09c5-437c-8ce5-865bab3c65c1
# ╠═72b7dc20-271d-4821-bfbe-410fe1b29f28
# ╟─e4d94be6-0027-4df9-bb37-b36dbe64ffa6
# ╠═7027a398-91f7-4d5f-860e-8e4242618b80
# ╠═4453e50e-0bf2-4cf3-a7e9-8ff5c3d034dc
# ╠═878f0a6a-43e1-4f23-9279-bedce534b576
# ╟─34f16752-e284-49ea-8335-16ba689ac924
# ╠═d6e56291-7b34-45e5-b7b7-e789086942cb
# ╠═e66874cf-245d-4211-a9fd-76ab6733c04b
# ╠═74b7e833-d38d-44f8-a248-cf0cfa9cac1a
# ╠═a9813130-40f0-4aec-9b19-0ed5d869ddd6
# ╠═bbee969f-6527-4590-ae7c-51ab5f132521
# ╠═278f4db4-eb6b-4aed-91bc-1dce3d3f612b
# ╟─987d86be-434d-47f7-9416-934ab13cf729
# ╠═de552769-e5e7-4a28-8693-6b783517b223
# ╠═fc01c02e-448c-43be-886d-788af2da8d26
# ╠═7a83109f-8354-46b2-8391-331a8d8e9285
# ╠═fc240a4f-b916-4e61-bb67-2ab45cfa533b
# ╠═f6533f60-a5f2-4bfc-a39a-2766e9ac3910
# ╠═fcafc53f-daaf-4831-a633-e5e2251ed02c
# ╠═2c3a835b-cb6f-4e04-945e-9b46e4cf3aa7
# ╠═bbfbf9fa-b0ff-4082-b5f7-ee70d7111f28
# ╠═a38bb098-0907-49d5-9861-6ecec8be130b
# ╠═7a0b949a-4ec0-4864-81d2-51828948b63c
# ╠═21e4f21f-4e9a-4c95-b4e8-a73c326c14ce
# ╟─d47100b1-6422-4585-9a2e-c2ef09fb9cf0
# ╠═8451bec4-0ae1-4766-ae71-ba997617b6e2
# ╠═5e0d9591-3f39-4ddf-a5be-c560af6def99
# ╠═8f13a36f-d326-4b63-a87c-d2f642750eef
# ╠═0e2268f3-18cd-4919-a90f-9e526589a8a7
# ╠═1e54d673-73c8-4389-aa49-6b031906900f
# ╠═1e8ad769-e259-4837-8cf8-b344ef994440
# ╠═35ba8c5b-4de6-4ae7-8af3-d06753d32f7e
