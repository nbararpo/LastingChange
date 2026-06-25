expr_matrix <- scale(expr_matrix)

protein_data <- data.frame(
   code               = data$code,
   time               = data$Time,
   sex                = as.factor(data$sex),
   age                = data$Age,
   score              = data$Score,
   protein_expression = expr_mat[, protein_name],
   stringsAsFactors   = FALSE
) %>%
   filter(complete.cases(.)) %>%
   mutate(
      # Ordered factor — adjust levels to your actual time labels
      time_ordered = factor(time,
         levels = c("C1", "C2", "C3", "C4", "C5", "C6"),
         ordered = TRUE
      )
   )

model <- clmm(
   time_ordered ~ score + protein_expression + sex + age +
      (1 | code),
   data = protein_data,
   link = "logit", # proportional odds
   Hess = TRUE # needed for SE extraction
)

coefs <- coef(summary(model))
