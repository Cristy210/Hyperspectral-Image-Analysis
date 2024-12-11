### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ d8a3a9ae-9e86-4b3d-88ec-b923691f4f97
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 2a46e3fd-09c5-437c-8ce5-865bab3c65c1
using LinearAlgebra, Dictionaries, MLDatasets, CairoMakie, Statistics

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

# ╔═╡ 987d86be-434d-47f7-9416-934ab13cf729


# ╔═╡ Cell order:
# ╟─22335c30-b725-11ef-2f34-6f691acbde73
# ╠═d8a3a9ae-9e86-4b3d-88ec-b923691f4f97
# ╠═2a46e3fd-09c5-437c-8ce5-865bab3c65c1
# ╠═72b7dc20-271d-4821-bfbe-410fe1b29f28
# ╠═7027a398-91f7-4d5f-860e-8e4242618b80
# ╠═4453e50e-0bf2-4cf3-a7e9-8ff5c3d034dc
# ╠═878f0a6a-43e1-4f23-9279-bedce534b576
# ╟─34f16752-e284-49ea-8335-16ba689ac924
# ╠═d6e56291-7b34-45e5-b7b7-e789086942cb
# ╠═e66874cf-245d-4211-a9fd-76ab6733c04b
# ╠═74b7e833-d38d-44f8-a248-cf0cfa9cac1a
# ╠═a9813130-40f0-4aec-9b19-0ed5d869ddd6
# ╠═bbee969f-6527-4590-ae7c-51ab5f132521
# ╠═987d86be-434d-47f7-9416-934ab13cf729
