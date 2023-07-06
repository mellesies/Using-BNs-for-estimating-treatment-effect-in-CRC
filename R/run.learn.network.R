run.learn.network <- function(
        df, variables, df.cardinality, layering=c(), algo="mmhc",
        title=NULL, title.variables=c(),
        display.plot = T, positions = NULL,
        save.plot=T, save.path=c('.'), filename.prefix="bnstruct - "

)
{
    # Create the dataset
    dataset <- create.dataset(df_coded, variables, df_cardinality)
    writeln("Created dataset ...")

    # if (length(layering)) {
    #     writeln("variables & layers:")
    #     print(data.frame(list(variables=variables, layering=layering)))
    # }

    # Generate title if not provided.
    if (is.null(title) || nchar(title) == 0) {
        if (length(title.variables) == 0) {
            title.variables = variables
        }

        title <- create.title(title.variables, algo, layering)
    }
    writeln("Using title:", title)


    # Learn a Bayesian Network, using constraints if provided
    write("Learning network ... ")
    net <- learn.network(
        dataset,
        layering=layering,
        algo=algo
    )
    writeln("[DONE]")

    if (algo == "mmpc") {
        bidirectional = TRUE
    } else {
        bidirectional = FALSE
    }

    # Plot to Jupyter
    if (display.plot) {
        writeln("Plotting ...")
        plot(
            net,
            method = "qgraph",
            title = title,
            layout = as.matrix(positions[variables, c("x", "y")]),
            groups = as.factor(positions[variables, "group"]),
            shape = "rectangle",
            vsize = 15,
            vsize2 = 5,
            color = brewer.pal(4, "Pastel2"),
            label.scale.equal = TRUE,
            legend = FALSE,
            bidirectional = bidirectional,
        )

        box(col = "#afafaf", which = "figure")
    }


    if (save.plot) {
        filename <- paste(filename.prefix, title, ".pdf", sep="")
        # file.path doesn't work well with vectors.
        # Need do.call to spread inputs.
        filename <- do.call(file.path, as.list(c(save.path, filename)))
        writeln("Saving plot to", filename)

        # Save to PDF
        pdf(filename, width=9, height=6)
        plot(
            net,
            method = "qgraph",
            title = title,
            layout = as.matrix(positions[variables, c("x", "y")]),
            groups = as.factor(positions[variables, "group"]),
            shape = "rectangle",
            vsize = 15,
            vsize2 = 5,
            color = brewer.pal(4, "Pastel2"),
            label.scale.equal = T,
            legend = F,
            bidirectional = bidirectional,
        )
        dev.off()


        # filename <- paste(filename.prefix, title, ".png", sep="")
        # # file.path doesn't work well with vectors. Need do.call to spread
        # # inputs.
        # filename <- do.call(file.path, as.list(c(save.path, filename)))
        # writeln("Saving plot to", filename)
        #
        # png(filename, width=800, height=600)
        # plot(
        #     net,
        #     method = "qgraph",
        #     title = title,
        #     layout = as.matrix(positions[variables, c("x", "y")]),
        #     groups = as.factor(positions[variables, "group"]),
        #     shape = "rectangle",
        #     vsize = 15,
        #     vsize2 = 5,
        #     color = brewer.pal(4, "Pastel2"),
        #     label.scale.equal = T,
        #     legend = F,
        # )
        # dev.off()

    }

    return(net)
}

