options(scipen=999)
options(digits=3)

setwd('C:/Users/lee/Documents')
fi <- read.csv("0831minmax.csv", fileEncoding="UTF-8-BOM")
str(fi)

pop <- aov(avg_sal_sum ~ pop, data =fi)
living_pop <- aov(avg_sal_sum ~ living_pop, data =fi)
floating_pop <- aov(avg_sal_sum ~ floating_pop, data =fi)
area_sum <- aov(avg_sal_sum ~ area_sum, data =fi)
review_visitor_sum <- aov(avg_sal_sum ~ review_visitor, data =fi)
review_blog_sum <- aov(avg_sal_sum ~ blog_visitor, data =fi)


summary(pop)
summary(living_pop)
summary(floating_pop)
summary(area_sum)
summary(review_visitor_sum)
summary(review_blog_sum)

