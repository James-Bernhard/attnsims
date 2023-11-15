#' Simulate observing a random vector multiple times
#'
#' This function simulates observing a `dim`-dimensional random vector
#' `n` times, whose components are independent, each of which can be
#' sampled from with the function `distribution` (a function of `n`).
#' @param n A positive integer giving the number of times to observe the
#' vector.
#' @param dim A positive integer giving the dimension of the random vector.
#' @param distribution_sampler A function of `n` that samples from the
#' distribution that each component of the vector is assumed to have.
#' @return An `n` by `dim` matrix whose rows are the observations of the
#' simulated random vector.
#' @export
observe_random_vector <- function(n = 1,
                                  dim = 1,
                                  distribution_sampler = stats::rnorm) {
  matrix(replicate(dim, distribution_sampler(n = n)), ncol = dim)
}


#' Compute softmax of a vector after dividing each component by something.
#'
#' This function takes a vector, divides all of its arguments by
#' `divided_by` (which defaults to 1), and then returns the softmax of the
#' resulting vector.
#' @param x A vector whose scaled softmax is to be computed.
#' @param divide_by A quantity to divide x by before applying softmax. This
#' is usually a scalar, but can in theory be a vector.
#' @return A vector containing the softmax of `x` divided by `divide_by`.
#' @export
scaled_softmax <- function(x, divide_by = 1) {
  exp(x / divide_by) / sum(exp(x / divide_by))
}


#' Simulate queries and keys and compute their attention scores
#'
#' This function computes a matrices of attention scores for simulated queries
#' and keys, one matrix for each function in the `distribution_samplers` list.
#'
#' @param distribution_sampler A function of the sample size `n` that
#' samples from the distribution that each component is to have.
#' @param n_keys A positive integer specifying the number of keys to simulate
#' @param n_queries A positive integer specifying the number of queries to
#' simulate.
#' @param key_dimension A positive integer specifying the number of
#' components each key and query is to have.
#' @param rescaling_function A function whose arguments are a matrix whose
#' rows are keys (`keys`) and a vector whose dimension is the number of
#' keys (`x`). This function is
#' applied to the keys and to the vector of attention scores
#' when computing the rescaled attention vector. For each set of keys, it is
#' a function from an n_key-dimensional space itself, although for some
#' types of simulations the codomain should be the standard
#' (n_key-1)-dimensional simplex in that space.
#' @param scalar_attention_function A function of two vectors (each of
#' dimension `key_dimension`) that returns a real number.
#' @return An `n_queries` by `n_keys` matrix whose (i,j) entry is the
#' scalar attention of the i-th query on the j-th key.
#' @export
simulate_attentions <-
  function(distribution_sampler = stats::rnorm,
           n_keys = 1,
           n_queries = 1,
           key_dimension = 1,
           rescaling_function = \(keys, x) x,
           scalar_attention_function = `%*%`)
  {
    compute_single_query_attentions <-
      function(q,
               keys,
               rescaling_function = \(keys, x) x,
               scalar_attention_function) {
        rescaling_function(keys = keys,
                           x = apply(keys, 1, function(k)
                             scalar_attention_function(q, k)))
      }

    queries <-
      observe_random_vector(n = n_queries,
                            dim = key_dimension,
                            distribution_sampler = distribution_sampler)
    keys <-
      observe_random_vector(n = n_keys,
                            dim = key_dimension,
                            distribution_sampler = distribution_sampler)

    output <- matrix(apply(queries, 1, function(q)
      compute_single_query_attentions(
        q,
        keys = keys,
        rescaling_function = rescaling_function,
        scalar_attention_function = scalar_attention_function
      )),
      nrow = n_queries)
    output
  }


#' Make a density plot of rescaled vector attention scores
#'
#' This function makes a kernel density estimate plot of the rescaled vector
#' attention scores for all the given queries and one or more keys (placed
#' into the rows of a matrix). By default only the first component of the
#' rescaled vector attention scores is plotted, but you can plot specific
#' key numbers by putting those numbers into the `key_index` vector. You can
#' also plot all components by setting `key_index` to `NULL`.
#'
#' @param attention_matrices An attention matrix or a list of attention
#' matrices, each of which will ordinarily be the output of
#' `simulate attentions`. That is, each attention matrix should have its
#' (i,j)-th entry be the scalar attention of the i-th query on the j-th key.
#' @param legend_texts A list (recycled if need be) of labels, one for each
#' attention matrix in `attention_matrices`. These labels will appear in the
#' legend of the plot to indicate which density plot is which.
#' @param scalar_attention_function A function of two variables (query and
#' key) that computes the scalar attention of the query on the key.
#' @param key_index The column numbers in the attention matrices that should
#' be included in the kernel density estimate plot.
#' @return A `ggplot2` kernel density estimate plot of attention scores.
#' @importFrom rlang .data
#' @export
plot_attentions <-
  function(attention_matrices,
           legend_texts = 1:length(attention_matrices),
           scalar_attention_function = `%*%`,
           key_index = 1) {
    if (!is.list(attention_matrices)) {
      attention_matrices <- list(attention_matrices)
    }

    n_plots <- length(attention_matrices)
    legend_texts <- rep(legend_texts, len = n_plots)
    attentions <- vector(mode = "list", length = n_plots)
    for (i in 1:n_plots) {
      if (is.null(key_index)) {
        key_index = 1:ncol(attention_matrices[[i]])
      }
      attentions[[i]] <-
        as.vector(attention_matrices[[i]][, key_index])
    }
    if (!is.null(legend_texts)) {
      names(attentions) <- legend_texts
    } else {
      names(attentions) <- 1:length(attentions)
    }
    df <- utils::stack(attentions)
    names(df) <- c("attention", "legend_text")
    output_plot <-
      ggplot2::ggplot(df,
                      ggplot2::aes(
                        x = rlang::.data$attention,
                        color = rlang::.data$legend_text
                      )) +
      ggplot2::geom_density() + ggplot2::geom_rug(alpha = 0.35) +
      ggplot2::xlab("Value") + ggplot2::ylab("Density")
    if (n_plots > 1) {
      output_plot <-
        ggplot2::ggplot(df,
                        ggplot2::aes(x = .data$attention, color = .data$legend_text))
    } else {
      output_plot <-
        ggplot2::ggplot(df, ggplot2::aes(x = .data$attention))
    }
    output_plot <-
      output_plot + ggplot2::geom_density() + ggplot2::geom_rug(alpha = 0.35) +
      ggplot2::xlab("Value") + ggplot2::ylab("Density")

    return(output_plot)
  }
