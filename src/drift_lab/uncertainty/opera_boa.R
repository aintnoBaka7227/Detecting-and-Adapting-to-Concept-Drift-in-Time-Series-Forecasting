# Line-delimited JSON server around opera's BOA, driven by opera_boa.py.
#
# One request per stdin line, one reply per stdout line. Each model is the
# authors' AgACI call (Zaffran et al. 2022, AgACI/Script/acp_gamma.R):
#
#   mixture(model = "BOA", loss.gradient = TRUE,
#           loss.type = list(name = "pinball", tau = tau))
#
# "update" feeds one block of expert forecasts and outcomes through
# predict(online = FALSE): opera's own BOA loop runs over the block's rows
# in order and the reply carries the coefficients left behind, which are
# the weights of the next issuance.

reply <- function(x) {
  cat(jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA), "\n", sep = "")
  flush(stdout())
}

started <- tryCatch({
  suppressPackageStartupMessages({
    library(opera)
    library(jsonlite)
  })
  list(ok = TRUE, opera = as.character(packageVersion("opera")), r = R.version.string)
}, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
reply(started)
if (!started$ok) quit(status = 1)

models <- list()
input <- file("stdin", "r")

while (length(line <- readLines(input, n = 1)) > 0) {
  out <- tryCatch({
    request <- fromJSON(line)
    key <- request$key
    if (request$op == "init") {
      models[[key]] <- mixture(model = "BOA", loss.gradient = TRUE,
                               loss.type = list(name = "pinball", tau = request$tau))
      list(ok = TRUE)
    } else if (request$op == "update") {
      y <- as.numeric(request$y)
      experts <- matrix(as.numeric(t(request$experts)), nrow = length(y), byrow = TRUE)
      model <- predict(models[[key]], newexperts = experts, newY = y,
                       online = FALSE, type = "model")
      # opera keeps the whole history on the object and re-binds it on every
      # call. BOA continues from `training` and `coefficients` alone, so drop it.
      model$Y <- NULL
      model$experts <- NULL
      model$awake <- NULL
      model$prediction <- NULL
      model$weights <- NULL
      models[[key]] <- model
      list(ok = TRUE, weights = as.numeric(model$coefficients))
    } else if (request$op == "drop") {
      models[[key]] <- NULL
      list(ok = TRUE)
    } else {
      stop(paste("unknown op", request$op))
    }
  }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  reply(out)
}
